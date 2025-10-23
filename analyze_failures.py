"""
Script to analyze qids where majority voting failed.

Reports qids where the majority answer is not correct, along with
metrics and number of sequences for each.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_voting_results(qid_folder: str) -> Dict[str, Any]:
    """Load voting results from qid folder."""
    results_path = os.path.join(qid_folder, 'voting_results.json')
    
    if not os.path.exists(results_path):
        return {}
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def find_qid_folders(directory: str) -> List[int]:
    """Scan directory and find all qidN folders."""
    qid_list = []
    directory_path = Path(directory)
    
    for item in directory_path.iterdir():
        if item.is_dir() and item.name.startswith('qid'):
            try:
                qid = int(item.name[3:])
                qid_list.append(qid)
            except ValueError:
                continue
    
    return sorted(qid_list)


def analyze_qid(qid: int, output_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """Analyze a single qid and check if majority voting failed.
    
    Returns:
        Tuple of (majority_failed, analysis_dict)
    """
    qid_folder = os.path.join(output_dir, f'qid{qid}')
    
    if not os.path.exists(qid_folder):
        return False, {}
    
    # Load voting results
    results = load_voting_results(qid_folder)
    
    if not results:
        return False, {}
    
    # Extract metadata
    meta = results.get('_meta', {})
    num_traces = meta.get('num_traces', 'N/A')
    correct_sampled = meta.get('correct_answer_sampled', False)
    ground_truth = meta.get('ground_truth', 'N/A')
    
    # Check majority voting
    majority_result = results.get('majority', {})
    majority_correct = majority_result.get('correct', False)
    majority_answer = majority_result.get('answer', 'N/A')
    
    # Count correct methods
    correct_methods = []
    total_methods = 0
    
    for method, result in results.items():
        if method == '_meta':
            continue
        total_methods += 1
        if result.get('correct', False):
            correct_methods.append(method)
    
    analysis = {
        'qid': qid,
        'ground_truth': ground_truth,
        'num_traces': num_traces,
        'correct_sampled': correct_sampled,
        'majority_answer': majority_answer,
        'majority_correct': majority_correct,
        'correct_methods': correct_methods,
        'num_correct_methods': len(correct_methods),
        'total_methods': total_methods,
        'all_results': results
    }
    
    return not majority_correct, analysis


def print_failures_table(failures: List[Dict[str, Any]]):
    """Print failures in a clean table format."""
    if not failures:
        return
    
    print(f"\nQIDs that failed maj@10:")
    print(f"{'='*80}")
    print(f"{'QID':>5} | {'Finished length':>15} | {'Correct sampled':>16} | {'Correct methods':>16}")
    print(f"{'-'*80}")
    
    for failure in failures:
        qid = failure['qid']
        num_traces = failure['num_traces']
        correct_sampled = '✓' if failure['correct_sampled'] else '✗'
        num_correct = failure['num_correct_methods']
        total_methods = failure['total_methods']
        correct_methods_str = f"{num_correct}/{total_methods}"
        
        print(f"{qid:>5} | {num_traces:>15} | {correct_sampled:>16} | {correct_methods_str:>16}")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze qids where majority voting failed'
    )
    parser.add_argument('--dir', type=str, required=True,
                       help='Directory containing qidN folders with voting_results.json')
    
    args = parser.parse_args()
    
    # Find all qid folders
    print(f"Scanning directory {args.dir}...")
    qid_list = find_qid_folders(args.dir)
    print(f"Found {len(qid_list)} qid folders")
    
    # Analyze each qid
    failures = []
    
    for qid in qid_list:
        majority_failed, analysis = analyze_qid(qid, args.dir)
        
        if majority_failed:
            failures.append(analysis)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"MAJORITY VOTING FAILURE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total QIDs analyzed:           {len(qid_list)}")
    print(f"QIDs where majority failed:    {len(failures)}")
    print(f"Majority voting success rate:  {100*(len(qid_list)-len(failures))/len(qid_list):.2f}%")
    
    if not failures:
        print("\n✓ No failures - majority voting succeeded on all questions!")
        return
    
    # Print table of failures
    print_failures_table(failures)
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'-'*80}")
    
    # Correct answer sampled in failures
    sampled_in_failures = sum(1 for f in failures if f['correct_sampled'])
    print(f"Failures where correct answer was sampled: {sampled_in_failures}/{len(failures)} "
          f"({100*sampled_in_failures/len(failures):.1f}%)")
    
    # Average number of traces in failures
    avg_traces = sum(f['num_traces'] for f in failures if isinstance(f['num_traces'], int)) / len(failures)
    print(f"Average sequences in failed cases:         {avg_traces:.1f}")
    
    # Average correct methods in failures
    avg_correct = sum(f['num_correct_methods'] for f in failures) / len(failures)
    avg_total = sum(f['total_methods'] for f in failures) / len(failures)
    print(f"Average correct methods in failures:       {avg_correct:.1f}/{avg_total:.1f} "
          f"({100*avg_correct/avg_total:.1f}%)")


if __name__ == '__main__':
    main()

