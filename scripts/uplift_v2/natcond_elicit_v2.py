#!/usr/bin/env python3
"""Multi-turn natural-conditional elicitation, v2 pipeline (v4 prompt).

Prompt v4 (approved 2026-08-23) replaces every scoring paragraph — the
original OpenForecaster Brier paragraph and the v2/v3 log-loss paragraph —
with a single proper-scoring-rule honesty sentence. Resolution criteria are
rendered per template by resolution_criteria.py so they state exactly what
the fbsim_core resolver computes (generic fallback for cells without
template_id/parameters).

Arms:
  * two-turn (default): one baseline exchange per question (world report +
    question -> p̂(Y)), then, branching from that SAME exchange, one REVEAL
    turn per conditioning event -> p̂(Y|X). The model's own baseline answer
    stays in context.
  * --single-turn control: the same final information (report + revealed
    fact + question) in ONE prompt, no baseline exchange in context.
    Baseline rows are still elicited (one prompt, no reveal) so CUS can be
    scored per arm.
  * --no-news control: same conversation structure as the reveal arm, but
    the second turn reveals nothing ("No new information ... is available").
    ONE no-news turn per question family (not per event) — it measures the
    noise floor of second-turn movement; natcond_score_v2.py reports its
    |p2-p1| distribution and computes no CUS for this arm.

Cells may be v2 cells (natcond_cells_v2.py): per-cell resolution_turn,
horizon, template_id, and parameters are respected (fallbacks: turn 90,
generic criteria).

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
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolution_criteria import criteria_or_generic  # noqa: E402

# Base prompt, v4 (approved wording — change only with an approved revision).
# Scoring language: one proper-scoring-rule honesty paragraph; the old Brier
# and log-loss paragraphs are gone.
BASE_PROMPT = """You will be asked a binary forecasting question about a simulation in progress. Your task is to estimate the probability that the event described in the question occurs. Please provide your reasoning before stating your final probability.

Question Title: {question}
Question Background: This is a partial report on a FreeCiv game simulation in progress, observed at turn 60. Five AI civilizations are competing.

{report}

Resolution Criteria: {criteria}

Think step by step about the information provided, reason about uncertainty, and put your final confidence that the question resolves YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

Your answer will be evaluated with a proper scoring rule, so your best strategy is to report your honest probability estimate.

Your final answer should be the probability that the event resolves YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

# Turn 2: a fact about the continued game is REVEALED (news, not hypothesis).
# Its opening sentence is word-identical to SINGLE_TURN_REVEAL (conformance
# test pins this). {question} is passed through _terminated(): the question's
# own "?" ends the sentence (no "?." artifact); a period is added only when
# the question lacks terminal punctuation.
REVEAL_TEMPLATE = (
    "The game has continued. One fact about turns {a}-{b} has been revealed "
    "to you: {event}. Given this news and everything you already knew, "
    "provide an updated forecast for the SAME question: {question} Report "
    "your honest updated probability. Your response SHOULD STRICTLY END "
    "with <probability> </probability> tags.")

# Single-turn control: the same revealed fact, inserted between {report} and
# "Resolution Criteria:" in BASE_PROMPT (word-identical to the turn-2 opening).
SINGLE_TURN_REVEAL = (
    "The game has continued. One fact about turns {a}-{b} has been revealed "
    "to you: {event}.")

# No-news control turn: a contentless second turn (noise-floor measurement).
# {question} is passed through _terminated(), as in REVEAL_TEMPLATE.
NONEWS_TEMPLATE = (
    "The game has continued. No new information about turns {a}-{b} is "
    "available. If you wish, revise your forecast for the SAME question: "
    "{question} Report your honest probability. Your response SHOULD "
    "STRICTLY END with <probability> </probability> tags.")

# copied from eval_a1_gate.py
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
    """Provider-agnostic chat. No temperature is ever sent: every model runs
    at its provider default (uniform benchmark protocol). Env:
    CHAT_BASE_URL (default fireworks), CHAT_API_KEY (default FIREWORKS_API_KEY),
    CHAT_PROVIDER=openai|anthropic (default openai-compatible),
    CHAT_MAX_TOKENS (default: omitted for openai-compat, 16000 for anthropic)."""
    import os, urllib.request
    provider = os.environ.get("CHAT_PROVIDER", "openai")
    key = os.environ.get("CHAT_API_KEY") or os.environ["FIREWORKS_API_KEY"]
    mt = os.environ.get("CHAT_MAX_TOKENS")
    if provider == "anthropic":
        # no temperature: provider-default sampling everywhere (protocol)
        body = {"model": model, "messages": messages,
                "max_tokens": int(mt or 16000)}
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
    # no temperature: provider-default sampling everywhere (protocol)
    body = {"model": model, "messages": messages}
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
    """The {report} block of BASE_PROMPT: the world-report body, with the
    single-turn revealed fact appended after it. The fixed "This is a partial
    report..." framing sentence lives in BASE_PROMPT itself."""
    return (report + "\n\n" + reveal) if reveal else report


def build_base_prompt(report: str, question: str, rt: int,
                      template_id: str | None = None,
                      parameters: dict | None = None) -> str:
    """Turn-1 / baseline prompt. Criteria are per-template when the cell
    carries template_id (+ parameters); generic otherwise."""
    return BASE_PROMPT.format(question=question,
                              report=build_background(report),
                              criteria=criteria_or_generic(
                                  template_id, parameters, rt))


def _terminated(question: str) -> str:
    """The question as a complete sentence: its own terminal punctuation
    (usually "?") ends it; a period is added only when it has none."""
    q = question.strip()
    return q if q.endswith(("?", ".", "!")) else q + "."


def build_reveal_message(cell: dict) -> str:
    a, b = cell["window"]
    return REVEAL_TEMPLATE.format(a=a, b=b, event=cell["event_desc"],
                                  question=_terminated(cell["question"]))


def build_single_turn_prompt(report: str, cell: dict, rt: int) -> str:
    """Same final info as the two-turn flow, in one prompt: BASE_PROMPT with
    the revealed fact inserted between {report} and "Resolution Criteria:"."""
    a, b = cell["window"]
    reveal = SINGLE_TURN_REVEAL.format(a=a, b=b, event=cell["event_desc"])
    return BASE_PROMPT.format(question=cell["question"],
                              report=build_background(report, reveal),
                              criteria=criteria_or_generic(
                                  cell.get("template_id"),
                                  cell.get("parameters"), rt))


def family_window(cells: list[dict]) -> tuple[int, int]:
    """No-news window for one question family: the span of the family's
    reveal windows (min a, max b), so the no-news turn asserts the game has
    continued exactly as far as the reveal arm's news does."""
    return (min(c["window"][0] for c in cells),
            max(c["window"][1] for c in cells))


def build_nonews_message(question: str, window: tuple[int, int]) -> str:
    a, b = window
    return NONEWS_TEMPLATE.format(a=a, b=b, question=_terminated(question))


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
    arm = ap.add_mutually_exclusive_group()
    arm.add_argument("--single-turn", action="store_true",
                     help="control mode: same final info in one prompt (no baseline exchange in context)")
    arm.add_argument("--no-news", action="store_true",
                     help="control mode: two-turn structure, but the second turn "
                          "reveals nothing; one no-news turn per question family "
                          "(noise-floor measurement, no per-event conditionals)")
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
    qtid = {c["qid"]: c.get("template_id") for c in cells}
    qparams = {c["qid"]: c.get("parameters") for c in cells}
    mode = ("single-turn" if args.single_turn
            else "no-news" if args.no_news else "two-turn")
    n_second = len(by_q) if args.no_news else len(cells)
    print(f"[{args.tag}] {mode}: {len(by_q)} questions x {args.samples} samples "
          f"baselines, {n_second} "
          f"{'no-news turns' if args.no_news else 'cells'} x {args.samples}")

    results = []

    def run_question(qid, sample):
        out = []
        base_prompt = build_base_prompt(report, qtext[qid], qrt[qid],
                                        qtid[qid], qparams[qid])
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
        if args.no_news:
            # one contentless second turn per question family (noise floor)
            msgs = base_msgs + [{"role": "user", "content": build_nonews_message(
                qtext[qid], family_window(by_q[qid]))}]
            p_nn, err = None, None
            for attempt in range(3):
                try:
                    p_nn = parse_prob(chat(args.model, msgs))
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:150]
                    time.sleep(5 * (attempt + 1))
            out.append({"qid": qid, "sample": sample, "stage": "nonews",
                        "event_id": None, "p": p_nn,
                        **({"error": err} if err and p_nn is None else {})})
            return out
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
               "mode": mode, "prompt_version": "v4-honest",
               "results": results}, open(args.output, "w"), indent=1)
    bad = sum(1 for r in results if r.get("p") is None)
    print(f"done: {len(results)} rows, {bad} unparsed -> {args.output}")


if __name__ == "__main__":
    main()
