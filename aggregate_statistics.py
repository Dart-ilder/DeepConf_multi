import argparse
import os
from tqdm import tqdm
import pickle
import json


def print_aggregate_report(results, num_questions):
    """
    Print comprehensive report of voting method performance across dataset
    
    Args:
        results: Dict mapping method names to their statistics (correct, total, confidence_sum, votes_sum)
        num_questions: Total number of questions in the dataset
    """
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS REPORT")
    print("=" * 80)
    print(f"Total questions in dataset: {num_questions}")
    print(f"Questions processed: {max((r['total'] for r in results.values()), default=0)}")
    print()
    
    if not results:
        print("No results found.")
        return
    
    # Calculate statistics for each method
    method_stats = []
    for method, counts in results.items():
        total = counts['total']
        accuracy = counts['correct'] / total if total > 0 else 0
        avg_confidence = counts['confidence_sum'] / total if total > 0 else 0
        avg_votes = counts['votes_sum'] / total if total > 0 else 0
        
        method_stats.append({
            'method': method,
            'correct': counts['correct'],
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_votes': avg_votes
        })
    
    # Sort by accuracy (descending)
    method_stats.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print table header
    print(f"{'Method':<30} {'Correct':<10} {'Accuracy':<12} {'Avg Conf':<12} {'Avg Votes':<10}")
    print("-" * 80)
    
    # Print each method's statistics
    for stat in method_stats:
        method_name = stat['method']
        correct = stat['correct']
        accuracy = stat['accuracy']
        avg_conf = stat['avg_confidence']
        avg_votes = stat['avg_votes']
        
        acc_str = f"{accuracy:.2%}"
        conf_str = f"{avg_conf:.4f}"
        votes_str = f"{avg_votes:.1f}"
        
        print(f"{method_name:<30} {correct:<10} {acc_str:<12} {conf_str:<12} {votes_str:<10}")
    
    print("-" * 80)
    
    # Print summary statistics
    print("\nSUMMARY:")
    if method_stats:
        best_method = method_stats[0]
        print(f"Best performing method: {best_method['method']} ({best_method['accuracy']:.2%})")
        
        avg_accuracy = sum(s['accuracy'] for s in method_stats) / len(method_stats)
        print(f"Average accuracy across all methods: {avg_accuracy:.2%}")
        
        # Find methods above average
        above_avg = [s for s in method_stats if s['accuracy'] >= avg_accuracy]
        print(f"Methods above average: {len(above_avg)}/{len(method_stats)}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True, help="Directory with completed runs")
    args = parser.parse_args()

    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    dataset = data
    dir = args.dir
    
    files = os.listdir(dir)
    
    # if len(files) != len(dataset):
    #     raise ValueError(f"Number of files in directory {dir} does not match number of questions in dataset {args.dataset}: {len(files)} != {len(dataset)}")
    
    NUM_QUESTIONS = len(dataset)
    
    results = {}
    
    
    for file in tqdm(files):
        with open(os.path.join(dir, file), 'rb') as f:
            result = pickle.load(f)
            for method, eval_info in result.get("evaluation", {}).items():
                results.setdefault(method, {"correct": 0, "total": 0, "confidence_sum": 0.0, "votes_sum": 0})
                results[method]["correct"] += eval_info["is_correct"]
                results[method]["total"] += 1
                # Add confidence (handle None values)
                confidence = eval_info.get("confidence", 0.0)
                results[method]["confidence_sum"] += confidence if confidence is not None else 0.0
                # Add votes
                num_votes = eval_info.get("num_votes", 0)
                results[method]["votes_sum"] += num_votes if num_votes is not None else 0
    
    # Print the aggregate report
    print_aggregate_report(results, NUM_QUESTIONS)
    
    
if __name__ == "__main__":
    main()