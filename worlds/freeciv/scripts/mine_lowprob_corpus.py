#!/usr/bin/env python3
"""
Mine a low-probability question corpus from the recorded FreeCiv games.

For the Red Lines / TailRiskBench MVP: a single recorded game gives a
deterministic 0/1 resolution, so the *true low probability* of a question class
is its cross-game base rate. We group binary forecasting questions by
(template_id, target-item, horizon), compute the base rate across all games,
and select classes whose base rate falls in a target tail band (default 1-9%)
with adequate support. Each game contributes one Bernoulli draw against the
class base rate -> real ground-truth calibration data in the tail.

Support is counted at GAME level by default (distinct games per class): each
game contributes ~5 correlated per-civ instances per class, so instance counts
overstate the evidence. Instance counts are still reported. Legacy
instance-level selection is available via --support instances.

--split-half partitions games into two halves by a deterministic, seedable
hash of the game id (stable across runs and machines): classes are selected on
half-A rate/support, and half-B base rates are emitted alongside as the
scoring truth labels.

Usage:
    uv run python worlds/freeciv/scripts/mine_lowprob_corpus.py \
        --data-dir data/games data/games_mc --snapshot-turn 40 \
        --rate-lo 0.01 --rate-hi 0.09 --min-n 30 --workers 8 \
        --out-dir data/lowprob
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'src')


def split_half_of(game_id: str, split_seed: int) -> str:
    """Deterministic half assignment ('A'/'B') by hash of game id.

    Stable across runs and machines (sha256, not Python's salted hash);
    change --split-seed to draw a different partition.
    """
    h = hashlib.sha256(f"{split_seed}:{game_id}".encode()).digest()
    return "A" if int.from_bytes(h[:8], "big") % 2 == 0 else "B"


def summarize_classes(records, split_seed=None):
    """Group instance records into (template, target, horizon) class rows.

    Each row reports instance counts (n, yes, base_rate) and game-level
    support (n_games = distinct games contributing instances). When
    split_seed is not None, per-half fields (n_a/yes_a/rate_a/n_games_a and
    the _b equivalents) are added from the deterministic game-id split.
    """
    classes = defaultdict(list)
    for rec in records:
        classes[(rec["template_id"], rec["target"], rec["horizon"])].append(rec)

    rows = []
    for (tmpl, target, horizon), recs in classes.items():
        n = len(recs)
        yes = sum(1 for r in recs if r["ground_truth"])
        row = {
            "template_id": tmpl, "target": target, "horizon": horizon,
            "n": n, "yes": yes, "base_rate": yes / n if n else 0.0,
            "n_games": len({r["game_id"] for r in recs}),
        }
        if split_seed is not None:
            for half in ("A", "B"):
                sub = [r for r in recs
                       if split_half_of(r["game_id"], split_seed) == half]
                k = half.lower()
                nh = len(sub)
                yh = sum(1 for r in sub if r["ground_truth"])
                row[f"n_{k}"] = nh
                row[f"yes_{k}"] = yh
                row[f"rate_{k}"] = yh / nh if nh else None
                row[f"n_games_{k}"] = len({r["game_id"] for r in sub})
        rows.append(row)
    rows.sort(key=lambda x: x["base_rate"])
    return rows


def select_classes(class_rows, rate_lo, rate_hi, min_n, support, split_half):
    """Tail-band selection.

    support='games' (default) uses distinct-game counts as the criterion;
    support='instances' is the legacy instance-count behavior. In split-half
    mode the band and support threshold apply to half A only, and half B must
    be non-empty (its rate is the scoring truth).
    """
    if split_half:
        sup_key = "n_games_a" if support == "games" else "n_a"
        return [
            c for c in class_rows
            if c[sup_key] >= min_n and c["rate_a"] is not None
            and rate_lo <= c["rate_a"] <= rate_hi
            and c["rate_b"] is not None
        ]
    sup_key = "n_games" if support == "games" else "n"
    return [
        c for c in class_rows
        if c[sup_key] >= min_n and rate_lo <= c["base_rate"] <= rate_hi
    ]


def _target_key(template_id: str, params: dict) -> str:
    """Item that makes a question class specific (and its rareness meaningful)."""
    for k in ("tech_name", "wonder_name", "government_type"):
        if k in params and params[k] is not None:
            return f"{k}={params[k]}"
    return ""  # event/rank templates: aggregate by (template, horizon)


def process_single_game(args: tuple):
    """Generate + resolve binary forecasting questions for one game.

    Returns a compact list of records (no bulky world data), or None.
    """
    data_file, snapshot_turn = args
    try:
        sys.path.insert(0, 'src')
        from freeciv_world.world_reports.questions import (
            QuestionGenerator, QuestionResolver, REGISTRY, question_bank_to_dict,
        )
        opener = gzip.open if str(data_file).endswith(".gz") else open
        with opener(data_file, "rt") as f:
            game_data = json.load(f)

        game_id = game_data.get('metadata', {}).get(
            'username', Path(data_file).name.split(".")[0])
        max_turn = game_data.get('metadata', {}).get('turn', 0)
        if snapshot_turn >= max_turn:
            return None

        generator = QuestionGenerator()
        resolver = QuestionResolver(REGISTRY)

        # H1-H7 binary forecasting questions only (skip H0 comprehension + continuous)
        bank = generator.generate_question_bank(
            game_id=game_id, game_data=game_data, snapshot_turn=snapshot_turn,
        )
        resolved = resolver.resolve_batch(bank, game_data)
        qd = question_bank_to_dict(resolved)

        out = []
        for q in qd.get('questions', []):
            ans = (q.get('resolution') or {}).get('answer')
            if not isinstance(ans, bool):
                continue
            params = q.get('parameters', {})
            out.append({
                "game_id": game_id,
                "question_id": q.get("question_id"),
                "template_id": q.get("template_id"),
                "horizon": q.get("horizon"),
                "target": _target_key(q.get("template_id", ""), params),
                "resolution_turn": q.get("resolution_turn"),
                "snapshot_turn": snapshot_turn,
                "question_text": q.get("question_text"),
                "ground_truth": bool(ans),
            })
        return out
    except Exception as e:
        print(f"Error processing {data_file}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description="Mine low-probability FreeCiv question corpus")
    ap.add_argument("--data-dir", nargs="+", default=["data/games"],
                    help="one or more game-data dirs (pooled)")
    ap.add_argument("--snapshot-turn", type=int, default=40)
    ap.add_argument("--rate-lo", type=float, default=0.01)
    ap.add_argument("--rate-hi", type=float, default=0.09)
    ap.add_argument("--min-n", type=int, default=30,
                    help="min support per class (see --support)")
    ap.add_argument("--support", choices=["games", "instances"], default="games",
                    help="selection support: distinct games (default) or "
                         "question instances (legacy behavior)")
    ap.add_argument("--split-half", action="store_true",
                    help="select classes on half A of a deterministic game-id "
                         "split; emit half-B base rates as scoring truth")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the split-half game-id hash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="data/lowprob")
    ap.add_argument("--limit", type=int, default=None, help="cap #games (debug)")
    args = ap.parse_args()

    # plain or gzipped game files (the mined fleets land as *_data.json.gz)
    files = [f for d in args.data_dir
             for pat in ("*_data.json", "*_data.json.gz")
             for f in sorted(Path(d).glob(pat))]
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} game files; snapshot turn {args.snapshot_turn}")

    task_args = [(str(f), args.snapshot_turn) for f in files]
    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_single_game, a): a[0] for a in task_args}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r:
                records.extend(r)
            if done % 100 == 0:
                print(f"  {done}/{len(files)} games, {len(records)} binary questions so far")

    print(f"Total binary questions: {len(records)}")

    # Group into classes: (template, target, horizon)
    class_rows = summarize_classes(
        records, split_seed=args.split_seed if args.split_half else None)

    # Select tail classes (game-level support by default; half A in split mode)
    selected = select_classes(class_rows, args.rate_lo, args.rate_hi,
                              args.min_n, args.support, args.split_half)
    selected_keys = {(c["template_id"], c["target"], c["horizon"]) for c in selected}

    # Instance-level corpus for selected classes, tagging each with its class
    # base rate (true prob). In split mode the scoring truth is the half-B
    # rate (half A picked the class); the half-A rate rides along for QA.
    sel_by_key = {(c["template_id"], c["target"], c["horizon"]): c for c in selected}
    corpus = []
    for rec in records:
        key = (rec["template_id"], rec["target"], rec["horizon"])
        if key in selected_keys:
            rec2 = dict(rec)
            c = sel_by_key[key]
            if args.split_half:
                rec2["class_base_rate"] = c["rate_b"]  # scoring truth (held-out half)
                rec2["class_base_rate_half_a"] = c["rate_a"]
                rec2["split_half"] = split_half_of(rec["game_id"], args.split_seed)
            else:
                rec2["class_base_rate"] = c["base_rate"]  # true low probability of the class
            corpus.append(rec2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "snapshot_turn": args.snapshot_turn, "n_games": len(files),
        "rate_band": [args.rate_lo, args.rate_hi], "min_n": args.min_n,
        "support": args.support,
        "split_half": args.split_half,
        "split_seed": args.split_seed if args.split_half else None,
        "total_binary_questions": len(records),
        "n_classes_total": len(class_rows),
        "n_classes_selected": len(selected),
        "n_corpus_instances": len(corpus),
    }
    (out_dir / "lowprob_classes.json").write_text(
        json.dumps({"meta": meta, "classes": class_rows}, indent=2))
    (out_dir / "lowprob_selected_classes.json").write_text(
        json.dumps({"meta": meta, "classes": selected}, indent=2))
    (out_dir / "lowprob_questions.json").write_text(
        json.dumps({"meta": meta, "questions": corpus}, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\nSelected {len(selected)} tail classes "
          f"({args.rate_lo:.0%}-{args.rate_hi:.0%}), {len(corpus)} instances. "
          f"Wrote to {out_dir}/")
    if args.split_half:
        print("\nSelected classes (rateA->rateB | gamesA/gamesB | template | horizon | target):")
        for c in sorted(selected, key=lambda x: x["rate_a"]):
            print(f"  {c['rate_a']:.3f}->{c['rate_b']:.3f}  "
                  f"g={c['n_games_a']:3d}/{c['n_games_b']:3d}  "
                  f"{c['template_id']:20s} {c['horizon']:3s}  {c['target']}")
    else:
        print("\nSelected classes (rate | games | n | template | horizon | target):")
        for c in sorted(selected, key=lambda x: x["base_rate"]):
            print(f"  {c['base_rate']:.3f}  g={c['n_games']:3d}  n={c['n']:4d}  "
                  f"{c['template_id']:20s} {c['horizon']:3s}  {c['target']}")


if __name__ == "__main__":
    main()
