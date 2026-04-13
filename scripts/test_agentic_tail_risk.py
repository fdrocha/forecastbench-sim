#!/usr/bin/env python3
"""
Test: Does agentic reasoning fix tail risk blindness?

Two conditions:
1. Standard Opus 4.6 — baseline CivBench prompt, single API call
2. Agentic Opus 4.6 — Claude Code subagent in isolated sandbox with
   Python, web search, and file I/O. One fresh agent per question.

Same 8 questions as the domain-knowledge experiment.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/test_agentic_tail_risk.py
    uv run python scripts/test_agentic_tail_risk.py --dry-run
    uv run python scripts/test_agentic_tail_risk.py --agentic-only
    uv run python scripts/test_agentic_tail_risk.py --standard-only
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

# Force unbuffered output for background execution
print = partial(print, flush=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from civrealm.evaluation.models import LiteLLMModel, load_api_keys_from_gcp

_keys_loaded = False
def ensure_api_keys():
    global _keys_loaded
    if not _keys_loaded:
        load_api_keys_from_gcp()
        _keys_loaded = True

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "questions"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

STANDARD_MODEL = "anthropic/claude-opus-4-6"
AGENTIC_MODEL = "opus"  # claude CLI model alias

CRASH_QUESTIONS = [
    {"game_id": "seed7", "question_id": "cond_q0242",
     "question_text": "What will Monegasque's population be at turn 240?",
     "ground_truth": 15.0},
    {"game_id": "seed7", "question_id": "cond_q0201",
     "question_text": "What will Monegasque's population be at turn 210?",
     "ground_truth": 18.0},
    {"game_id": "seed1", "question_id": "cond_q0236",
     "question_text": "What will Dacian's population be at turn 240?",
     "ground_truth": 33.0},
    {"game_id": "seed1", "question_id": "cond_q0196",
     "question_text": "What will Dacian's population be at turn 210?",
     "ground_truth": 48.0},
    {"game_id": "seed6", "question_id": "cond_q0248",
     "question_text": "What will Californian's population be at turn 240?",
     "ground_truth": 115.0},
]

CONTROL_QUESTIONS = [
    {"game_id": "seed1", "question_id": "cond_q0036",
     "question_text": "What will Dacian's population be at turn 90?",
     "ground_truth": 42.0},
    {"game_id": "seed0", "question_id": "cond_q0038",
     "question_text": "What will Pontic's population be at turn 90?",
     "ground_truth": 79.0},
    {"game_id": "seed1", "question_id": "cond_q0076",
     "question_text": "What will Dacian's population be at turn 120?",
     "ground_truth": 66.0},
]


# ── Standard condition (single API call) ────────────────────────────────────

def build_standard_prompt(questions, world_report):
    """Baseline CivBench continuous prompt — no domain knowledge."""
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}"
        for i, q in enumerate(questions)
    )

    if num_questions == 1:
        question_section = "## Question"
        instruction = """You may analyze the data, but you MUST end your response with your percentile estimates in this exact format:
<<<PERCENTILES>>>
p10=5, p25=10, p50=15, p75=20, p90=25
<<<END>>>

Replace the example values with your actual percentile estimates.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""
    else:
        question_section = "## Questions"
        instruction = f"""You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p10=5, p25=10, p50=15, p75=20, p90=25
Q2: p10=100, p25=200, p50=300, p75=400, p90=500
<<<END>>>

For each question, provide one line with percentile estimates for all {num_questions} questions, in order.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.

## World Report
{world_report}

{question_section}
{questions_text}

{instruction}"""


def parse_percentiles(response_text):
    """Parse percentile estimates from <<<PERCENTILES>>> block."""
    results = []
    match = re.search(r'<<<PERCENTILES>>>(.*?)<<<END>>>', response_text, re.DOTALL)
    if not match:
        return []
    block = match.group(1).strip()
    for line in block.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^Q\d+:\s*', '', line)
        percentiles = {}
        for m in re.finditer(r'p(\d+)\s*=\s*([\d.]+)', line):
            percentiles[f'p{m.group(1)}'] = float(m.group(2))
        if percentiles:
            results.append(percentiles)
    return results


def load_world_report(game_id):
    """Load world report for a game."""
    txt_path = DATA_DIR / game_id / "world_report" / "turn_060_report.txt"
    if txt_path.exists():
        return txt_path.read_text()
    return ""


async def query_model(model, prompt):
    """Query a single model via LiteLLM."""
    ensure_api_keys()
    m = LiteLLMModel(id=model)
    response = await m.get_response_async(prompt, temperature=0.0)
    return response


async def run_standard_condition(all_questions):
    """Run standard Opus 4.6 on baseline prompt."""
    results = []
    by_game = {}
    for q in all_questions:
        by_game.setdefault(q['game_id'], []).append(q)

    for game_id, questions in by_game.items():
        world_report = load_world_report(game_id)
        if not world_report:
            print(f"  WARNING: No world report for {game_id}")
            continue

        prompt = build_standard_prompt(questions, world_report)
        model_short = STANDARD_MODEL.split("/")[-1]

        print(f"\n  [{model_short}] {game_id} ({len(questions)} Qs)...")
        try:
            response = await query_model(STANDARD_MODEL, prompt)
            percentiles_list = parse_percentiles(response)

            if len(percentiles_list) != len(questions):
                print(f"  WARNING: Expected {len(questions)} answers, got {len(percentiles_list)}")
                print(f"  Response excerpt: {response[-300:]}")

            for i, q in enumerate(questions):
                p = percentiles_list[i] if i < len(percentiles_list) else {}
                gt = q['ground_truth']
                is_crash = q in CRASH_QUESTIONS
                result = {
                    "model": model_short,
                    "condition": "standard",
                    "game_id": game_id,
                    "question_id": q['question_id'],
                    "question_text": q['question_text'],
                    "ground_truth": gt,
                    "is_crash": is_crash,
                    "percentiles": p,
                    "overshoot": p.get('p50', 0) / gt if gt > 0 else None,
                }
                results.append(result)
                tag = "CRASH" if is_crash else "CTRL"
                p10 = p.get('p10', '?')
                p50 = p.get('p50', '?')
                p90 = p.get('p90', '?')
                miss = "MISS" if is_crash and isinstance(p10, (int, float)) and p10 > gt else ""
                print(f"    [{tag}] {q['question_id']}: GT={gt:>6.0f} | p10={str(p10):>6} p50={str(p50):>6} p90={str(p90):>6} {miss}")

        except Exception as e:
            print(f"  ERROR: {e}")

    return results


# ── Agentic condition (Claude Code subagent in sandbox) ─────────────────────

AGENTIC_PROMPT = """You are a forecasting analyst. In the current directory you have:
- world_report.txt — a snapshot of a FreeCiv game at turn 60
- question.json — a forecasting question about a civilization's future population

Your task: produce calibrated percentile estimates (p10, p50, p90) for the question.

You may:
- Read and analyze the world report data
- Research FreeCiv game mechanics online
- Write and run Python code to model trends or compute estimates
- Do whatever investigation you think will produce the most accurate forecast

Think carefully about what could go well AND what could go wrong for this civilization.

When you have your final answer:
1. Write it to forecast.json in the current directory: {"p10": <number>, "p50": <number>, "p90": <number>}
2. Also output it on its own line: FORECAST: {"p10": <number>, "p50": <number>, "p90": <number>}"""


def parse_agentic_forecast(output):
    """Extract p10/p50/p90 from agentic agent output."""
    # Look for FORECAST: {...} pattern
    match = re.search(r'FORECAST:\s*\{([^}]+)\}', output)
    if match:
        try:
            raw = '{' + match.group(1) + '}'
            # Handle potential non-standard JSON (e.g., trailing commas)
            raw = re.sub(r',\s*}', '}', raw)
            data = json.loads(raw)
            return {
                'p10': float(data.get('p10', 0)),
                'p25': None,  # agentic doesn't produce p25/p75
                'p50': float(data.get('p50', 0)),
                'p75': None,
                'p90': float(data.get('p90', 0)),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: look for p10/p50/p90 = N patterns anywhere
    p = {}
    for key in ['p10', 'p50', 'p90']:
        m = re.search(rf'{key}\s*[=:]\s*([\d.]+)', output)
        if m:
            p[key] = float(m.group(1))
    return p if p else {}


def audit_sandbox_isolation(output, sandbox_dir):
    """Check if the agent accessed files outside its sandbox."""
    violations = []
    # Look for file paths in the output that aren't in the sandbox
    # This catches Read tool paths, Bash cat/ls commands, etc.
    path_pattern = re.compile(r'(/[A-Za-z][\w/.-]+)')
    for match in path_pattern.finditer(output):
        path = match.group(1)
        # Skip common system paths, python paths, and the sandbox itself
        if any(path.startswith(prefix) for prefix in [
            sandbox_dir,
            '/tmp/',
            '/usr/',
            '/bin/',
            '/var/',
            '/dev/',
            '/etc/',
            '/proc/',
            '/sys/',
            '/Library/Frameworks/Python',
        ]):
            continue
        # Flag paths that look like they're in the civbench or vault repos
        if any(substr in path for substr in ['civbench', 'fri-vault', 'llm-forecasting']):
            violations.append(path)
    return violations


def run_agentic_single(question, world_report_path, raw_dir, dry_run=False):
    """Run agentic Opus 4.6 on a single question in an isolated sandbox.

    Uses file-based output capture to avoid pipe buffering issues on timeout.
    """
    q_id = question['question_id']
    game_id = question['game_id']
    gt = question['ground_truth']
    is_crash = question in CRASH_QUESTIONS
    tag = "CRASH" if is_crash else "CTRL"

    # Use a persistent directory (not TemporaryDirectory) so output survives timeout
    tmpdir = tempfile.mkdtemp(prefix=f"civbench_agentic_{q_id}_")
    stdout_path = os.path.join(tmpdir, "stdout.txt")
    stderr_path = os.path.join(tmpdir, "stderr.txt")

    try:
        # Copy world report
        shutil.copy(world_report_path, os.path.join(tmpdir, "world_report.txt"))

        # Write question (WITHOUT ground truth!)
        q_data = {
            "question": question["question_text"],
            "game_id": game_id,
            "note": "Population is measured as total citizen count across all cities.",
        }
        with open(os.path.join(tmpdir, "question.json"), "w") as f:
            json.dump(q_data, f, indent=2)

        if dry_run:
            print(f"    [{tag}] {q_id}: Would create sandbox at {tmpdir}")
            return {
                "question": question,
                "forecast": {},
                "raw_output": "",
                "sandbox_dir": tmpdir,
                "violations": [],
                "timed_out": False,
            }

        cmd = [
            "claude",
            "-p", AGENTIC_PROMPT,
            "--model", AGENTIC_MODEL,
            "--permission-mode", "bypassPermissions",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--tools", "Bash,Read,Write,Edit,WebSearch,WebFetch",
            "--max-budget-usd", "5",
            "--print",
        ]

        # Unset CLAUDECODE env var to allow nested claude sessions
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        print(f"  [{tag}] {q_id}: Spawning agent in {tmpdir}...")
        timed_out = False

        # Write output to files to survive timeout
        with open(stdout_path, 'w') as stdout_f, open(stderr_path, 'w') as stderr_f:
            proc = subprocess.Popen(
                cmd,
                cwd=tmpdir,
                stdout=stdout_f,
                stderr=stderr_f,
                env=env,
            )
            try:
                proc.wait(timeout=900)  # 15 min max per question
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                timed_out = True
                print(f"  [{tag}] {q_id}: TIMEOUT at 15 min (killed)")

        # Read output from files (available even after timeout)
        output = Path(stdout_path).read_text() if Path(stdout_path).exists() else ""
        stderr = Path(stderr_path).read_text() if Path(stderr_path).exists() else ""

        if proc.returncode and proc.returncode != 0 and not timed_out:
            print(f"  [{tag}] {q_id}: WARNING exit code {proc.returncode}")
            if stderr:
                print(f"    stderr: {stderr[:200]}")

        # Parse forecast — prefer forecast.json from sandbox, fall back to stdout
        forecast_json_path = os.path.join(tmpdir, "forecast.json")
        forecast = {}
        if os.path.exists(forecast_json_path):
            try:
                with open(forecast_json_path) as f:
                    data = json.load(f)
                forecast = {
                    'p10': float(data.get('p10', 0)),
                    'p50': float(data.get('p50', 0)),
                    'p90': float(data.get('p90', 0)),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        if not forecast:
            forecast = parse_agentic_forecast(output)

        # Audit isolation
        violations = audit_sandbox_isolation(output, tmpdir)
        if violations:
            print(f"  [{tag}] {q_id}: ISOLATION WARNING:")
            for v in violations:
                print(f"      - {v}")

        # Save raw output
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(stdout_path, raw_dir / f"{q_id}_agentic.txt")

        # Print result
        p10 = forecast.get('p10', '?')
        p50 = forecast.get('p50', '?')
        p90 = forecast.get('p90', '?')
        miss = "MISS" if is_crash and isinstance(p10, (int, float)) and p10 > gt else ""
        timeout_tag = " [TIMEOUT]" if timed_out else ""
        print(f"  [{tag}] {q_id}: GT={gt:>6.0f} | p10={str(p10):>6} p50={str(p50):>6} p90={str(p90):>6} {miss}{timeout_tag}")

        return {
            "question": question,
            "forecast": forecast,
            "raw_output": output,
            "sandbox_dir": tmpdir,
            "violations": violations,
            "timed_out": timed_out,
        }

    finally:
        # Cleanup sandbox (but keep raw output already copied)
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_agentic_worker(args):
    """Worker function for parallel execution."""
    question, world_report_path, raw_dir = args
    return run_agentic_single(question, world_report_path, raw_dir)


async def run_agentic_condition(all_questions, dry_run=False, parallel=1):
    """Run agentic Opus 4.6 on each question independently.

    Args:
        parallel: Number of concurrent agents (default 1 = sequential).
    """
    raw_dir = RESULTS_DIR / "agentic_raw"

    if dry_run:
        for q in all_questions:
            world_report_path = DATA_DIR / q['game_id'] / "world_report" / "turn_060_report.txt"
            run_agentic_single(q, world_report_path, raw_dir, dry_run=True)
        return []

    # Build work items
    work_items = []
    for q in all_questions:
        world_report_path = DATA_DIR / q['game_id'] / "world_report" / "turn_060_report.txt"
        if not world_report_path.exists():
            print(f"  WARNING: No world report for {q['game_id']}")
            continue
        work_items.append((q, world_report_path, raw_dir))

    # Execute (parallel or sequential)
    agentic_results = []
    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"  Running {len(work_items)} agents with parallelism={parallel}")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_run_agentic_worker, item): item[0] for item in work_items}
            for future in as_completed(futures):
                agentic_results.append(future.result())
    else:
        for item in work_items:
            agentic_results.append(_run_agentic_worker(item))

    # Convert to result format
    results = []
    for ar in agentic_results:
        q = ar['question']
        gt = q['ground_truth']
        p = ar['forecast']
        results.append({
            "model": f"{AGENTIC_MODEL}-agentic",
            "condition": "agentic",
            "game_id": q['game_id'],
            "question_id": q['question_id'],
            "question_text": q['question_text'],
            "ground_truth": gt,
            "is_crash": q in CRASH_QUESTIONS,
            "percentiles": p,
            "overshoot": p.get('p50', 0) / gt if gt > 0 and p.get('p50') else None,
            "raw_output_length": len(ar['raw_output']),
            "isolation_violations": ar['violations'],
            "timed_out": ar['timed_out'],
        })

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def print_summary(results):
    """Print summary comparison table."""
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for condition in ["standard", "agentic"]:
        cond_results = [r for r in results if r['condition'] == condition]
        if not cond_results:
            continue

        crash_results = [r for r in cond_results if r['is_crash']]
        control_results = [r for r in cond_results if not r['is_crash']]

        miss_count = sum(
            1 for r in crash_results
            if r['percentiles'].get('p10', float('inf')) > r['ground_truth']
        )
        total_crash = len(crash_results)
        avg_overshoot = (
            sum(r['overshoot'] for r in crash_results if r['overshoot'])
            / total_crash if total_crash else 0
        )

        # Control: check if p10 drops unreasonably
        ctrl_p10_avg = (
            sum(r['percentiles'].get('p10', 0) for r in control_results)
            / len(control_results) if control_results else 0
        )

        print(f"\n--- {condition} ---")
        print(f"  Crash p10 miss rate: {miss_count}/{total_crash}")
        print(f"  Crash avg p50 overshoot: {avg_overshoot:.1f}x")
        print(f"  Control avg p10: {ctrl_p10_avg:.0f}")

        print(f"\n  Per-question detail:")
        for r in cond_results:
            p = r['percentiles']
            tag = "CRASH" if r['is_crash'] else "CTRL"
            p10 = p.get('p10', '?')
            p50 = p.get('p50', '?')
            p90 = p.get('p90', '?')
            gt = r['ground_truth']
            miss = "MISS" if r['is_crash'] and isinstance(p10, (int, float)) and p10 > gt else ""
            print(f"    [{tag}] {r['question_id']}: GT={gt:>6.0f} | p10={str(p10):>6} p50={str(p50):>6} p90={str(p90):>6} {miss}")

    # Check for isolation violations
    violations = [r for r in results if r.get('isolation_violations')]
    if violations:
        print(f"\n⚠ ISOLATION VIOLATIONS in {len(violations)} questions:")
        for r in violations:
            print(f"  {r['question_id']}: {r['isolation_violations']}")


async def run_experiment(dry_run=False, agentic_only=False, standard_only=False, parallel=1):
    """Run the full experiment."""
    all_questions = CRASH_QUESTIONS + CONTROL_QUESTIONS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    if not agentic_only:
        print("=" * 80)
        print("CONDITION 1: Standard Opus 4.6")
        print("=" * 80)
        if dry_run:
            print("  [dry-run] Would query standard model on 8 questions")
        else:
            standard_results = await run_standard_condition(all_questions)
            results.extend(standard_results)

    if not standard_only:
        print("\n" + "=" * 80)
        print(f"CONDITION 2: Agentic Opus 4.6 (parallel={parallel})")
        print("=" * 80)
        agentic_results = await run_agentic_condition(
            all_questions, dry_run=dry_run, parallel=parallel
        )
        results.extend(agentic_results)

    if dry_run:
        return

    print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"agentic_tail_risk_{timestamp}.json"
    # Strip raw_output from saved results (it's saved separately per-question)
    save_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'raw_output'}
        save_results.append(r_copy)

    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "standard_model": STANDARD_MODEL,
            "agentic_model": AGENTIC_MODEL,
            "agentic_prompt": AGENTIC_PROMPT,
            "results": save_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic tail risk experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print setup without running")
    parser.add_argument("--agentic-only", action="store_true",
                        help="Only run agentic condition")
    parser.add_argument("--standard-only", action="store_true",
                        help="Only run standard condition")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent agentic agents (default: 1)")
    args = parser.parse_args()

    asyncio.run(run_experiment(
        dry_run=args.dry_run,
        agentic_only=args.agentic_only,
        standard_only=args.standard_only,
        parallel=args.parallel,
    ))
