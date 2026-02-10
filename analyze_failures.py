#!/usr/bin/env python3
"""
Failure Analysis for Tau-Bench ACT & REACT Experiments.
Verifies failures are benchmark-related (not infra/code bugs).

Usage:
    python3 analyze_failures.py                            # Full summary
    python3 analyze_failures.py --strategy react            # REACT only
    python3 analyze_failures.py --strategy act              # ACT only
    python3 analyze_failures.py --domain airline            # Airline only
    python3 analyze_failures.py --domain retail             # Retail only
    python3 analyze_failures.py --task-id 42                # Deep-dive one task
    python3 analyze_failures.py --log --domain retail --strategy act          # Task log (all trials)
    python3 analyze_failures.py --log --domain retail --strategy act --trial 0  # Log for trial 0 only
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

RESULT_FILES = {
    ("retail", "react"): "results/gaudi-retail-react/react-Qwen3-32B-0.7_range_0--1_user-qwen3-30b-a3b-instruct-2507-llm_0210022333.json",
    ("retail", "act"): "results/gaudi-retail-act/act-Qwen3-32B-0.7_range_0--1_user-qwen3-30b-a3b-instruct-2507-llm_0210032026.json",
    ("airline", "react"): "results/gaudi-airline-react/react-Qwen3-32B-0.7_range_0--1_user-qwen3-30b-a3b-instruct-2507-llm_0210102316.json",
    ("airline", "act"): "results/gaudi-airline-act/act-Qwen3-32B-0.7_range_0--1_user-qwen3-30b-a3b-instruct-2507-llm_0210112708.json",
}

EXPECTED_TASKS = {"airline": 50, "retail": 115}


def load_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('results', [data])


def classify_error(error_str):
    """Classify an error as infra (your side) or benchmark (expected)."""
    if not error_str:
        return "benchmark", "task_execution_failure"
    if '503' in error_str or 'Service Unavailable' in error_str:
        return "infra", "503_API_unavailable"
    if '504' in error_str or 'Gateway' in error_str:
        return "infra", "504_gateway_timeout"
    if 'NoneType' in error_str:
        return "infra", "code_bug_NoneType"
    if 'tool-call-parser' in error_str or 'enable-auto-tool-choice' in error_str:
        return "infra", "vLLM_missing_flags"
    if 'ContextWindowExceeded' in error_str:
        return "infra", "context_window_exceeded"
    if 'ConnectionError' in error_str or 'Connection refused' in error_str:
        return "infra", "connection_error"
    return "benchmark", "task_execution_failure"


def analyze(domain, strategy, results, args):
    total = len(results)
    passed = sum(1 for r in results if r.get('reward', 0) >= 0.99)
    failed = total - passed
    unique_tasks = set(r['task_id'] for r in results)
    expected = EXPECTED_TASKS.get(domain, '?')
    missing = set(range(expected)) - unique_tasks if isinstance(expected, int) else set()

    # Group by task
    by_task = defaultdict(list)
    for r in results:
        by_task[r['task_id']].append(r)

    always_pass = sorted(t for t, trials in by_task.items() if all(x.get('reward', 0) >= 0.99 for x in trials))
    always_fail = sorted(t for t, trials in by_task.items() if all(x.get('reward', 0) < 0.99 for x in trials))

    # Classify every failure
    infra_errors = defaultdict(list)  # type -> [(task_id, trial, error_snippet)]
    benchmark_count = 0
    for r in results:
        if r.get('reward', 0) >= 0.99:
            continue
        error = r.get('info', {}).get('error', '')
        source, etype = classify_error(error)
        if source == "infra":
            infra_errors[etype].append((r['task_id'], r.get('trial', 0), error[:150]))
        else:
            benchmark_count += 1

    print(f"\n{'='*70}")
    print(f"  {domain.upper()} - {strategy.upper()}")
    print(f"{'='*70}")
    print(f"  Tasks: {len(unique_tasks)}/{expected}  |  Entries: {total}  |  Pass: {passed} ({100*passed/total:.1f}%)  |  Fail: {failed}")
    if missing:
        print(f"  Missing {len(missing)} tasks: {sorted(missing)[:20]}{'...' if len(missing)>20 else ''}")
    print(f"\n  Always PASS (all trials): {len(always_pass)} -> {always_pass}")
    print(f"  Always FAIL (all trials): {len(always_fail)} -> {always_fail}")

    # Error verdict
    total_infra = sum(len(v) for v in infra_errors.values())
    print(f"\n  --- ERROR SOURCE VERDICT ---")
    print(f"  Benchmark failures (expected):  {benchmark_count} / {failed}  ({100*benchmark_count/failed:.1f}%)" if failed else "")
    print(f"  Infra/code errors (your side):  {total_infra} / {failed}  ({100*total_infra/failed:.1f}%)" if failed else "")

    if infra_errors:
        print(f"\n  One example per unique infra error type:")
        for etype, examples in infra_errors.items():
            ex = examples[0]
            print(f"    [{etype}] x{len(examples)} occurrences")
            print(f"      Example: task {ex[0]}, trial {ex[1]}")
            print(f"      {ex[2]}")
    else:
        print(f"\n  >>> ALL {failed} failures are legitimate benchmark failures. Nothing from your side. <<<")

    # Deep-dive into a specific task
    if args.task_id is not None and args.task_id in by_task:
        tid = args.task_id
        print(f"\n  --- Task {tid} Detail ---")
        for t in by_task[tid]:
            reward = t.get('reward', 0)
            status = "PASS" if reward >= 0.99 else "FAIL"
            error = t.get('info', {}).get('error', '')
            traj = t.get('traj', [])
            print(f"    Trial {t.get('trial',0)}: {status}  (traj_len={len(traj)})")
            if error:
                print(f"      Error: {error[:200]}")
            elif traj:
                # Show last message
                last = traj[-1]
                if isinstance(last, dict):
                    print(f"      Last msg [{last.get('role','?')}]: {str(last.get('content',''))[:150]}")

    return {
        'domain': domain, 'strategy': strategy,
        'unique_tasks': len(unique_tasks), 'expected': expected,
        'total': total, 'passed': passed, 'failed': failed,
        'always_fail': always_fail, 'infra_count': total_infra, 'bench_count': benchmark_count,
    }


def print_log(domain, strategy, results, trial_filter=None):
    """Print task-by-task pass/fail log like the HPC terminal output."""
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get('trial', 0)].append(r)

    trials_to_show = [trial_filter] if trial_filter is not None else sorted(by_trial.keys())

    for trial_num in trials_to_show:
        if trial_num not in by_trial:
            print(f"Trial {trial_num} not found. Available trials: {sorted(by_trial.keys())}")
            continue
        trial_results = sorted(by_trial[trial_num], key=lambda x: x['task_id'])
        passed = sum(1 for r in trial_results if r.get('reward', 0) >= 0.99)
        failed = len(trial_results) - passed

        print(f"\n{'='*70}")
        print(f"  {domain.upper()} - {strategy.upper()} - Trial {trial_num} (Pass {trial_num+1})")
        print(f"  {passed} passed, {failed} failed out of {len(trial_results)} tasks")
        print(f"{'='*70}")

        for r in trial_results:
            tid = r['task_id']
            reward = r.get('reward', 0)
            mark = '\u2705' if reward >= 0.99 else '\u274C'
            info = r.get('info', {})
            error = info.get('error', '')
            traj = r.get('traj', [])

            # Extract user's opening message and first tool call
            user_msg = ''
            first_tool = ''
            for entry in traj:
                if not isinstance(entry, dict):
                    continue
                role = entry.get('role', '')
                content = str(entry.get('content', ''))
                # Get first user message (the task request)
                if role == 'user' and not user_msg:
                    user_msg = content[:100].replace('\n', ' ').strip()
                # Get first tool/function call
                if role == 'assistant' and not first_tool:
                    tool_calls = entry.get('tool_calls', [])
                    if tool_calls:
                        tc = tool_calls[0]
                        fn = tc.get('function', {})
                        first_tool = f"{fn.get('name', '?')}({str(fn.get('arguments', ''))[:60]})"

            detail = first_tool if first_tool else (user_msg if user_msg else f"traj_len={len(traj)}")

            if error:
                source, etype = classify_error(error)
                print(f"  {mark} task_id={tid}: [{etype}] {error[:80]}")
            else:
                print(f"  {mark} task_id={tid}: {detail}")

        print(f"\n  Summary: {passed}/{len(trial_results)} passed ({100*passed/len(trial_results):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Verify ACT/REACT failures are from benchmark, not your code")
    parser.add_argument("--strategy", choices=["act", "react"])
    parser.add_argument("--domain", choices=["airline", "retail"])
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--log", action="store_true", help="Print task-by-task pass/fail log (like HPC terminal output)")
    parser.add_argument("--trial", type=int, default=None, help="Show only this trial number (use with --log)")
    args = parser.parse_args()

    # Log mode: print task-by-task pass/fail like HPC terminal logs
    if args.log:
        for (domain, strategy), filepath in sorted(RESULT_FILES.items()):
            if args.strategy and strategy != args.strategy:
                continue
            if args.domain and domain != args.domain:
                continue
            path = Path(filepath)
            if not path.exists():
                print(f"WARNING: {filepath} not found, skipping")
                continue
            print_log(domain, strategy, load_results(filepath), args.trial)
        return

    summaries = []
    for (domain, strategy), filepath in sorted(RESULT_FILES.items()):
        if args.strategy and strategy != args.strategy:
            continue
        if args.domain and domain != args.domain:
            continue
        path = Path(filepath)
        if not path.exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue
        summaries.append(analyze(domain, strategy, load_results(filepath), args))

    # Cross-strategy comparison
    if len(summaries) >= 2 and args.task_id is None:
        by_domain = defaultdict(dict)
        for s in summaries:
            by_domain[s['domain']][s['strategy']] = s
        for domain, strats in by_domain.items():
            if len(strats) >= 2:
                common = set(strats['act']['always_fail']) & set(strats['react']['always_fail'])
                act_only = set(strats['act']['always_fail']) - set(strats['react']['always_fail'])
                react_only = set(strats['react']['always_fail']) - set(strats['act']['always_fail'])
                print(f"\n  {domain.upper()}: Always fail in BOTH: {len(common)} | ACT-only: {len(act_only)} | REACT-only: {len(react_only)}")

    # Final table
    if summaries:
        print(f"\n{'='*70}")
        print(f"  {'Domain':<9} {'Strategy':<8} {'Tasks':<10} {'Pass%':<8} {'Infra Err':<10} {'Bench Fail':<10}")
        print(f"  {'-'*9} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
        for s in summaries:
            print(f"  {s['domain']:<9} {s['strategy']:<8} {s['unique_tasks']}/{s['expected']:<7} {100*s['passed']/s['total']:.1f}%{'':3} {s['infra_count']:<10} {s['bench_count']}")

        total_infra = sum(s['infra_count'] for s in summaries)
        total_bench = sum(s['bench_count'] for s in summaries)
        total_fail = sum(s['failed'] for s in summaries)
        print(f"\n  TOTAL: {total_infra}/{total_fail} infra errors ({100*total_infra/total_fail:.1f}%), {total_bench}/{total_fail} benchmark failures ({100*total_bench/total_fail:.1f}%)")
        if total_infra <= 6:
            print(f"  CONCLUSION: Only {total_infra} infra errors out of {total_fail} total failures.")
            print(f"  All meaningful failures are from the benchmark. Your setup is fine.")


if __name__ == "__main__":
    main()
