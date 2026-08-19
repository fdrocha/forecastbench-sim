#!/usr/bin/env python3
"""Multi-turn natural-conditional elicitation, v2 (REVEAL wording).

Differences from natcond_elicit.py (v1):
  * Turn 1 uses excess-log-loss scoring language (the true probability over
    re-runs of the world is known; the perfect forecaster sets the floor and
    the penalty is the excess log loss above it), replacing the Brier
    paragraph of the OpenForecaster prompt. The rest of the prompt skeleton
    is unchanged.
  * Turn 2 REVEALS a fact about the continued game ("The game has continued.
    One fact about turns a-b has been revealed to you: ...") instead of the
    v1 "suppose that" hypothetical, and restates the question.
  * --single-turn control mode: the same final information (report + revealed
    fact + question + scoring language) in ONE prompt, no baseline exchange
    in context. Baseline rows are still elicited (one prompt, no reveal) so
    CUS can be scored per arm.
  * Cells may be v2 cells (natcond_cells_v2.py): per-cell resolution_turn and
    horizon are respected (fallback: turn 90).

Per question (per sample), two-turn mode: one baseline exchange
  user: OF binary prompt (world report + question)  ->  assistant: p̂(Y)
then, branching from that SAME exchange, one second turn per conditioning
event -> p̂(Y|X). The model's own baseline answer stays in context.

Usage:
  set -a; source .env; set +a
  uv run python scripts/uplift_v2/natcond_elicit_v2.py \
      --cells tmp/natcond_v2/w01_cells.json --game-id w01 \
      --model accounts/fireworks/models/gpt-oss-120b --tag oss120 \
      --samples 2 --concurrency 12 --output tmp/natcond_v2/elicit_w01_oss120.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
from collections import defaultdict
from pathlib import Path

# OpenForecaster binary prompt skeleton (build_of_dataset.OF_BINARY_PROMPT)
# with the Brier scoring paragraph replaced by excess-log-loss scoring: the
# truth is a probability (many re-runs of the world), and the penalty is the
# excess log loss above the perfect forecaster's floor.
OF_BINARY_PROMPT_V2 = """You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).

Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

Estimate the probability of YES as accurately as you can. We have re-run this world many times, so the true probability of this event is known. Your answer is scored by log loss against that true probability: a perfect forecaster (one that reports the true probability) sets the floor, and your penalty is your excess log loss above it. Getting the odds right matters — saying 1% when the truth is 10% costs far more than saying 30% when the truth is 39%. Do not report 0 or 1.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

# Turn 2: a fact about the continued game is REVEALED (news, not hypothesis).
REVEAL_TEMPLATE = (
    "The game has continued. One fact about turns {a}-{b} has been revealed "
    "to you: {event}. Given this news and everything you already knew, "
    "provide an updated forecast for the SAME question: {question} "
    "The same scoring applies: your excess log loss against the true "
    "probability given this information. End with your updated probability "
    "in <probability> </probability> tags.")

# Single-turn control: same revealed fact folded into the one-shot background.
SINGLE_TURN_REVEAL = (
    "The game has continued. One fact about turns {a}-{b} has been revealed "
    "to you: {event}.")

# copied from eval_a1_gate.py (self-contained: no uplift_v2 imports on main)
PROB_RE = re.compile(r"<probability>\s*([0-9.eE+-]+)\s*</probability>")


def parse_prob(text: str) -> float | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = PROB_RE.findall(text)
    if not m:
        return None
    try:
        p = float(m[-1])
    except ValueError:
        return None
    if -0.01 <= p <= 1.01:
        return min(1.0, max(0.0, p))
    return None


def chat(model, messages):
    """Provider-agnostic chat. Env:
    CHAT_BASE_URL (default fireworks), CHAT_API_KEY (default FIREWORKS_API_KEY),
    CHAT_PROVIDER=openai|anthropic (default openai-compatible),
    CHAT_MAX_TOKENS (default: omitted for openai-compat, 16000 for anthropic)."""
    import os, urllib.request
    provider = os.environ.get("CHAT_PROVIDER", "openai")
    key = os.environ.get("CHAT_API_KEY") or os.environ["FIREWORKS_API_KEY"]
    mt = os.environ.get("CHAT_MAX_TOKENS")
    if provider == "anthropic":
        body = {"model": model, "messages": messages,
                "max_tokens": int(mt or 16000), "temperature": 1.0}
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json",
                     "User-Agent": "civbench-natcond/2.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        return "".join(b.get("text", "") for b in d.get("content", []))
    base = os.environ.get("CHAT_BASE_URL",
                          "https://api.fireworks.ai/inference/v1")
    body = {"model": model, "messages": messages, "temperature": 1.0}
    if mt:
        body["max_tokens"] = int(mt)
    elif "fireworks" in base:
        body["max_tokens"] = 8192  # fireworks default is tiny; keep old behavior
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "civbench-natcond/2.0"})
    # exponential backoff: high concurrency self-throttles on 429/5xx
    import random as _rnd
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < 6:
                time.sleep(min(120, 2 ** attempt) + _rnd.random() * 2)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 6:
                time.sleep(min(120, 2 ** attempt) + _rnd.random() * 2)
                continue
            raise


def build_background(report: str, reveal: str | None = None) -> str:
    background = ("This is a partial report on a FreeCiv game simulation in "
                  "progress, observed at turn 60. Five AI civilizations are "
                  "competing.\n\n" + report)
    if reveal:
        background += "\n\n" + reveal
    return background


def build_base_prompt(report: str, question: str, rt: int) -> str:
    rc = (f"Resolves YES if the answer to the question is affirmative in the "
          f"simulation state at turn {rt}, as determined by the game's "
          f"recorded metrics.")
    return OF_BINARY_PROMPT_V2.format(question_title=question,
                                      background=build_background(report),
                                      resolution_criteria=rc)


def build_reveal_message(cell: dict) -> str:
    a, b = cell["window"]
    return REVEAL_TEMPLATE.format(a=a, b=b, event=cell["event_desc"],
                                  question=cell["question"])


def build_single_turn_prompt(report: str, cell: dict, rt: int) -> str:
    """Same final info as the two-turn flow, in one prompt."""
    a, b = cell["window"]
    reveal = SINGLE_TURN_REVEAL.format(a=a, b=b, event=cell["event_desc"])
    rc = (f"Resolves YES if the answer to the question is affirmative in the "
          f"simulation state at turn {rt}, as determined by the game's "
          f"recorded metrics.")
    return OF_BINARY_PROMPT_V2.format(question_title=cell["question"],
                                      background=build_background(report, reveal),
                                      resolution_criteria=rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--preamble", default=None, help="file with rules/scaffold text prepended to the background")
    ap.add_argument("--report", default=None,
                    help="world report txt (default: data/questions_mc/<game-id>/world_report/turn_060_report.txt)")
    ap.add_argument("--single-turn", action="store_true",
                    help="control mode: same final info in one prompt (no baseline exchange in context)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cells = json.load(open(args.cells))
    report_path = args.report or (f"data/questions_mc/{args.game_id}/"
                                  f"world_report/turn_060_report.txt")
    report = Path(report_path).read_text()
    report = re.sub(r"^={10,} WORLD REPORT.*?\n|={10,} END REPORT.*$", "",
                    report, flags=re.DOTALL).strip()
    if args.preamble:
        report = Path(args.preamble).read_text().strip() + "\n\n" + report

    by_q = defaultdict(list)
    for c in cells:
        by_q[c["qid"]].append(c)
    qtext = {c["qid"]: c["question"] for c in cells}
    qrt = {c["qid"]: c.get("resolution_turn") or 90 for c in cells}
    mode = "single-turn" if args.single_turn else "two-turn"
    print(f"[{args.tag}] {mode}: {len(by_q)} questions x {args.samples} samples "
          f"baselines, {len(cells)} cells x {args.samples} conditionals")

    results = []

    def run_question(qid, sample):
        out = []
        base_prompt = build_base_prompt(report, qtext[qid], qrt[qid])
        for attempt in range(3):
            try:
                base_ans = chat(args.model, [
                    {"role": "user", "content": base_prompt}])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return [{"qid": qid, "sample": sample, "stage": "base",
                             "p": None, "error": str(e)[:150]}]
                time.sleep(5 * (attempt + 1))
        p_base = parse_prob(base_ans)
        out.append({"qid": qid, "sample": sample, "stage": "base",
                    "event_id": None, "p": p_base})
        base_msgs = [{"role": "user", "content": base_prompt},
                     {"role": "assistant", "content": base_ans}]
        for c in by_q[qid]:
            if args.single_turn:
                msgs = [{"role": "user",
                         "content": build_single_turn_prompt(report, c, qrt[qid])}]
            else:
                msgs = base_msgs + [{"role": "user",
                                     "content": build_reveal_message(c)}]
            p_cond, err = None, None
            for attempt in range(3):
                try:
                    ans = chat(args.model, msgs)
                    p_cond = parse_prob(ans)
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:150]
                    time.sleep(5 * (attempt + 1))
            out.append({"qid": qid, "sample": sample, "stage": "cond",
                        "event_id": c["event_id"], "p": p_cond,
                        **({"error": err} if err and p_cond is None else {})})
        return out

    jobs = [(q, s) for q in by_q for s in range(args.samples)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_question, q, s) for q, s in jobs]
        for k, f in enumerate(cf.as_completed(futs)):
            results.extend(f.result())
            print(f"  question-block {k+1}/{len(jobs)} done "
                  f"({time.time()-t0:.0f}s, {len(results)} rows)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "tag": args.tag, "cells": args.cells,
               "mode": mode, "prompt_version": "v2-reveal",
               "results": results}, open(args.output, "w"), indent=1)
    bad = sum(1 for r in results if r.get("p") is None)
    print(f"done: {len(results)} rows, {bad} unparsed -> {args.output}")


if __name__ == "__main__":
    main()
