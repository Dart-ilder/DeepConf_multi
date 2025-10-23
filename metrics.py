"""
Script to calculate voting metrics and accuracy for generated traces.

This script processes trace data from qidN folders, computes voting results
using various strategies, and calculates accuracy metrics.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from deepconf.utils import compute_all_voting_results


def load_dataset(dataset_path: str) -> Dict[int, str]:
    """Load dataset and return mapping of qid to ground truth answer."""
    qid_to_answer = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            qid_to_answer[idx] = str(data.get('answer', '')).strip()
    return qid_to_answer


def find_qid_folders(directory: str) -> List[int]:
    """Scan directory and find all qidN folders, returning list of qid integers."""
    qid_list = []
    directory_path = Path(directory)
    
    for item in directory_path.iterdir():
        if item.is_dir() and item.name.startswith('qid'):
            try:
                qid = int(item.name[3:])  # Extract number after 'qid'
                qid_list.append(qid)
            except ValueError:
                continue
    
    return sorted(qid_list)


def load_traces(qid_folder: str) -> List[Dict[str, Any]]:
    """Load traces from qid folder."""
    traces_path = os.path.join(qid_folder, 'traces.json')
    
    if not os.path.exists(traces_path):
        print(f"Warning: traces.json not found in {qid_folder}")
        return []
    
    with open(traces_path, 'r') as f:
        traces = json.load(f)
    
    return traces


def load_judge_traces(judge_folder: str) -> List[Dict[str, Any]]:
    """Load judge traces from judge folder."""
    return load_traces(judge_folder)


def merge_judge_confidences(traces: List[Dict[str, Any]], 
                            judge_traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge judge confidences into main traces.
    
    Assumes traces and judge_traces are aligned by index.
    """
    merged_traces = []
    
    for i, trace in enumerate(traces):
        merged_trace = trace.copy()
        
        # Add judge confidences if available
        if i < len(judge_traces) and 'confs' in judge_traces[i]:
            merged_trace['judge_confs'] = judge_traces[i]['confs']
        
        merged_traces.append(merged_trace)
    
    return merged_traces


def filter_length_stopped_traces(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out traces that stopped due to length."""
    return [trace for trace in traces if trace.get('stop_reason') != 'length']


def save_voting_results(results: Dict[str, Any], output_path: str):
    """Save voting results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def check_answer_match(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False
    
    # Normalize strings for comparison
    pred_norm = str(predicted).strip().lower()
    gt_norm = str(ground_truth).strip().lower()
    
    return pred_norm == gt_norm


def process_qid(qid: int, 
                output_dir: str, 
                ground_truth: str,
                judge_dir: Optional[str] = None) -> Dict[str, Any]:
    """Process a single qid folder.
    
    Returns dict with voting results and correctness for each method.
    """
    qid_folder = os.path.join(output_dir, f'qid{qid}')
    
    if not os.path.exists(qid_folder):
        print(f"Warning: Folder {qid_folder} not found")
        return {}
    
    # Load main traces
    traces = load_traces(qid_folder)
    
    if not traces:
        return {}
    
    print(f"Processing qid{qid}: Loaded {len(traces)} traces")
    
    # Filter out length-stopped traces
    traces = filter_length_stopped_traces(traces)
    print(f"  After filtering length-stopped: {len(traces)} traces")
    
    if not traces:
        print(f"  No valid traces remaining for qid{qid}")
        return {}
    
    # Check if correct answer was sampled at least once
    sampled_answers = [trace.get('extracted_answer') for trace in traces 
                      if trace.get('extracted_answer') is not None]
    correct_sampled = any(check_answer_match(ans, ground_truth) for ans in sampled_answers)
    
    print(f"  Ground truth: {ground_truth}")
    print(f"  Correct answer sampled: {correct_sampled}")
    
    # Load and merge judge confidences if provided
    if judge_dir:
        judge_qid_folder = os.path.join(judge_dir, f'qid{qid}')
        if os.path.exists(judge_qid_folder):
            judge_traces = load_judge_traces(judge_qid_folder)
            judge_traces = filter_length_stopped_traces(judge_traces)
            
            if len(judge_traces) == len(traces):
                traces = merge_judge_confidences(traces, judge_traces)
                print(f"  Merged judge confidences from {judge_qid_folder}")
            else:
                print(f"  Warning: Judge trace count ({len(judge_traces)}) doesn't match main trace count ({len(traces)})")
    
    # Compute voting results
    voting_results = compute_all_voting_results(traces)
    
    # Add correctness information
    results_with_correctness = {}
    correct_methods = []
    
    for method, result in voting_results.items():
        if result and result.get('answer') is not None:
            is_correct = check_answer_match(result['answer'], ground_truth)
            results_with_correctness[method] = {
                **result,
                'correct': is_correct,
                'ground_truth': ground_truth
            }
            if is_correct:
                correct_methods.append(method)
        else:
            results_with_correctness[method] = {
                'answer': None,
                'correct': False,
                'ground_truth': ground_truth
            }
    
    # Add sampling metric
    results_with_correctness['_meta'] = {
        'correct_answer_sampled': correct_sampled,
        'ground_truth': ground_truth,
        'num_traces': len(traces)
    }
    
    # Report voting results
    print(f"\n  Voting Results:")
    print(f"  {'-'*50}")
    
    # Sort methods by correctness first, then by name
    sorted_methods = sorted(
        [(m, r) for m, r in results_with_correctness.items() if m != '_meta'],
        key=lambda x: (not x[1].get('correct', False), x[0])
    )
    
    for method, result in sorted_methods:
        answer = result.get('answer', 'None')
        is_correct = result.get('correct', False)
        status = '✓' if is_correct else '✗'
        print(f"  {status} {method:40s}: {answer}")
    
    print(f"  {'-'*50}")
    print(f"  Correct methods: {len(correct_methods)}/{len(voting_results)}")
    
    # Save results
    results_path = os.path.join(qid_folder, 'voting_results.json')
    save_voting_results(results_with_correctness, results_path)
    print(f"  Saved results to {results_path}")
    
    return results_with_correctness


def calculate_overall_accuracy(all_results: Dict[int, Dict[str, Any]]) -> tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Calculate accuracy for each voting method across all qids.
    
    Returns tuple of (accuracy_results, sampling_stats).
    """
    
    # Collect all voting methods (excluding _meta)
    all_methods = set()
    for qid_results in all_results.values():
        all_methods.update(k for k in qid_results.keys() if k != '_meta')
    
    # Calculate accuracy for each method
    accuracy_results = {}
    
    for method in all_methods:
        correct_count = 0
        total_count = 0
        
        for qid, qid_results in all_results.items():
            if method in qid_results:
                result = qid_results[method]
                if result.get('answer') is not None:
                    total_count += 1
                    if result.get('correct', False):
                        correct_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            accuracy_results[method] = {
                'accuracy': round(accuracy, 4),
                'correct': correct_count,
                'total': total_count
            }
        else:
            accuracy_results[method] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0
            }
    
    # Calculate sampling statistics
    correct_sampled_count = 0
    total_qids = 0
    
    for qid, qid_results in all_results.items():
        if '_meta' in qid_results:
            total_qids += 1
            if qid_results['_meta'].get('correct_answer_sampled', False):
                correct_sampled_count += 1
    
    sampling_stats = {
        'correct_answer_sampled': {
            'count': correct_sampled_count,
            'total': total_qids,
            'rate': round(correct_sampled_count / total_qids, 4) if total_qids > 0 else 0.0
        }
    }
    
    return accuracy_results, sampling_stats


def main():
    parser = argparse.ArgumentParser(
        description='Calculate voting metrics and accuracy for generated traces'
    )
    parser.add_argument('--dir', type=str, required=True,
                       help='Directory containing qidN folders')
    parser.add_argument('--dataset', type=str, required=False, default="aime_2025.jsonl",
                       help='Path to dataset JSONL file')
    parser.add_argument('--judge_dir', type=str, default=None,
                       help='Optional directory containing judge model traces')
    
    args = parser.parse_args()
    
    # Load dataset ground truth
    print(f"Loading dataset from {args.dataset}...")
    qid_to_answer = load_dataset(args.dataset)
    print(f"Loaded {len(qid_to_answer)} questions")
    
    # Find all qid folders
    print(f"\nScanning directory {args.dir}...")
    qid_list = find_qid_folders(args.dir)
    print(f"Found {len(qid_list)} qid folders: {qid_list}")
    
    # Process each qid
    all_results = {}
    
    for qid in qid_list:
        if qid not in qid_to_answer:
            print(f"\nWarning: qid{qid} not found in dataset, skipping")
            continue
        
        print(f"\n{'='*60}")
        ground_truth = qid_to_answer[qid]
        results = process_qid(qid, args.dir, ground_truth, args.judge_dir)
        
        if results:
            all_results[qid] = results
    
    # Calculate overall accuracy
    print(f"\n{'='*60}")
    print("Calculating overall accuracy...")
    accuracy_results, sampling_stats = calculate_overall_accuracy(all_results)
    
    # Print sampling statistics
    print(f"\n{'='*60}")
    print("SAMPLING STATISTICS")
    print(f"{'='*60}")
    
    sampled_count = sampling_stats['correct_answer_sampled']['count']
    sampled_total = sampling_stats['correct_answer_sampled']['total']
    sampled_rate = sampling_stats['correct_answer_sampled']['rate']
    
    print(f"Correct answer sampled at least once: {sampled_count}/{sampled_total} ({sampled_rate:.2%})")
    
    # Print accuracy results
    print(f"\n{'='*60}")
    print("ACCURACY RESULTS")
    print(f"{'='*60}")
    
    # Sort by accuracy
    sorted_methods = sorted(accuracy_results.items(), 
                           key=lambda x: x[1]['accuracy'], 
                           reverse=True)
    
    for method, metrics in sorted_methods:
        acc = metrics['accuracy']
        correct = metrics['correct']
        total = metrics['total']
        print(f"{method:45s}: {acc:.2%} ({correct}/{total})")
    
    # Save overall results to file
    overall_results = {
        'accuracy': accuracy_results,
        'sampling': sampling_stats
    }
    
    accuracy_file = os.path.join(args.dir, 'overall_accuracy.json')
    with open(accuracy_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nOverall results saved to {accuracy_file}")


if __name__ == '__main__':
    main()

