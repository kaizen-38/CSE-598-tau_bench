#!/usr/bin/env python3
"""
Result Aggregator for Tau-Bench
Merges results from multiple JSON checkpoint files, handles duplicates, and generates final metrics.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set
import argparse


def load_json_results(filepath: str) -> List[Dict[str, Any]]:
    """Load results from a JSON checkpoint file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'results' in data:
                return data['results']
            else:
                return [data]
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}", file=sys.stderr)
        return []


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate results based on (task_id, trial) tuple.
    Keeps the most recent result if duplicates exist.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}
    
    for result in results:
        task_id = result.get('task_id')
        trial = result.get('trial', 0)
        key = (task_id, trial)
        
        # Keep the most recent (last one wins)
        seen[key] = result
    
    return list(seen.values())


def aggregate_results(result_dirs: List[str], output_file: str) -> None:
    """Aggregate all results from multiple directories into a single file."""
    all_results: List[Dict[str, Any]] = []
    
    print(f"Scanning {len(result_dirs)} result directories...")
    
    for result_dir in result_dirs:
        result_path = Path(result_dir)
        if not result_path.exists():
            print(f"Warning: {result_dir} does not exist, skipping", file=sys.stderr)
            continue
        
        # Find all JSON files
        json_files = list(result_path.glob("*.json"))
        print(f"  {result_dir}: Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            results = load_json_results(str(json_file))
            all_results.extend(results)
            print(f"    {json_file.name}: {len(results)} results")
    
    print(f"\nTotal results before deduplication: {len(all_results)}")
    
    # Deduplicate
    unique_results = deduplicate_results(all_results)
    print(f"Total results after deduplication: {len(unique_results)}")
    
    # Group by task_id and trial
    by_task_trial: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for result in unique_results:
        task_id = result.get('task_id')
        trial = result.get('trial', 0)
        by_task_trial[(task_id, trial)].append(result)
    
    # Check for duplicates (shouldn't happen after deduplication, but verify)
    duplicates = {k: v for k, v in by_task_trial.items() if len(v) > 1}
    if duplicates:
        print(f"Warning: Found {len(duplicates)} duplicate (task_id, trial) pairs after deduplication!", file=sys.stderr)
        for key, vals in list(duplicates.items())[:5]:
            print(f"  {key}: {len(vals)} entries", file=sys.stderr)
    
    # Save aggregated results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(unique_results, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total unique (task_id, trial) pairs: {len(by_task_trial)}")
    
    # Count by trial
    by_trial: Dict[int, int] = defaultdict(int)
    for (task_id, trial), _ in by_task_trial.items():
        by_trial[trial] += 1
    
    print("\nResults by trial:")
    for trial in sorted(by_trial.keys()):
        print(f"  Trial {trial}: {by_trial[trial]} tasks")
    
    # Count successes
    successes = sum(1 for r in unique_results if r.get('reward', 0) >= 0.99)
    failures = len(unique_results) - successes
    print(f"\nSuccess rate: {successes}/{len(unique_results)} = {100*successes/len(unique_results):.1f}%")
    print(f"  ✅ Successes: {successes}")
    print(f"  ❌ Failures: {failures}")
    
    # Check completeness
    max_trial = max(by_trial.keys()) if by_trial else 0
    tasks_per_trial = len(set(task_id for (task_id, _) in by_task_trial.keys()))
    expected_total = tasks_per_trial * (max_trial + 1)
    print(f"\nCompleteness: {len(unique_results)}/{expected_total} = {100*len(unique_results)/expected_total:.1f}%")
    print(f"  Expected: {tasks_per_trial} tasks × {max_trial + 1} trials = {expected_total}")


def find_missing_tasks(result_dirs: List[str], domain: str, num_trials: int = 5) -> Dict[int, Set[int]]:
    """
    Find missing (task_id, trial) combinations.
    Returns: Dict[trial, Set[task_id]]
    """
    all_results: List[Dict[str, Any]] = []
    
    for result_dir in result_dirs:
        result_path = Path(result_dir)
        if not result_path.exists():
            continue
        
        for json_file in result_path.glob("*.json"):
            results = load_json_results(str(json_file))
            all_results.extend(results)
    
    unique_results = deduplicate_results(all_results)
    
    # Get expected task IDs
    if domain == "airline":
        expected_task_ids = set(range(50))
    elif domain == "retail":
        expected_task_ids = set(range(115))
    else:
        print(f"Warning: Unknown domain {domain}, cannot determine expected task IDs", file=sys.stderr)
        return {}
    
    # Find missing
    completed: Set[tuple] = set()
    for result in unique_results:
        task_id = result.get('task_id')
        trial = result.get('trial', 0)
        if task_id is not None:
            completed.add((task_id, trial))
    
    missing: Dict[int, Set[int]] = defaultdict(set)
    for trial in range(num_trials):
        for task_id in expected_task_ids:
            if (task_id, trial) not in completed:
                missing[trial].add(task_id)
    
    return missing


def main():
    parser = argparse.ArgumentParser(description="Aggregate tau-bench results from multiple runs")
    parser.add_argument("result_dirs", nargs="+", help="Result directories to aggregate")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing tasks")
    parser.add_argument("--domain", choices=["airline", "retail"], help="Domain for missing task check")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of trials (for missing check)")
    
    args = parser.parse_args()
    
    aggregate_results(args.result_dirs, args.output)
    
    if args.check_missing and args.domain:
        print("\n=== Missing Tasks Check ===")
        missing = find_missing_tasks(args.result_dirs, args.domain, args.num_trials)
        
        total_missing = sum(len(task_ids) for task_ids in missing.values())
        if total_missing == 0:
            print("✅ All tasks completed!")
        else:
            print(f"Missing {total_missing} (task_id, trial) combinations:")
            for trial in sorted(missing.keys()):
                task_ids = sorted(missing[trial])
                print(f"  Trial {trial}: {len(task_ids)} missing tasks")
                if len(task_ids) <= 20:
                    print(f"    Task IDs: {task_ids}")
                else:
                    print(f"    Task IDs: {task_ids[:10]} ... {task_ids[-10:]}")


if __name__ == "__main__":
    main()
