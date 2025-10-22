"""
Utility functions for DeepThinkLLM

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    return None



def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs


def compute_least_grouped(confs: List[float], group_size: int) -> List[float]:
    """Compute sliding window mean confidence"""
    # Use numpy convolution for efficient sliding-window mean computation
    if not confs:
        return [0]

    arr = np.array(confs, dtype=float)
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    if arr.size < group_size:
        return [float(np.round(np.mean(arr), 3))]

    # moving average via convolution
    kernel = np.ones(group_size, dtype=float)
    window_sums = np.convolve(arr, kernel, mode='valid')
    sliding = np.round(window_sums / group_size, 3).tolist()
    return sliding


# ============= VOTING FUNCTIONS =============

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from confs in a trace"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence_from_confs(confs: List[float], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens given a conf array."""
    if not confs:
        return 0.0
    tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    return float(np.mean(tail_confs)) if tail_confs else 0.0


def calculate_bottom_window_confidence(trace: Dict[str, Any], window_size: int = 2048, bottom_percent: float = 0.1) -> float:
    """Calculate mean confidence from sliding windows, return average of bottom percentile"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            if len(confs) < window_size:
                return np.mean(confs)
            
            window_means = []
            current_sum = sum(confs[:window_size])
            window_means.append(current_sum / window_size)
            
            for i in range(1, len(confs) - window_size + 1):
                current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
                window_means.append(current_sum / window_size)
            
            if not window_means:
                return 0.0
            
            if bottom_percent == -1:  # Min window
                return min(window_means)
            
            num_bottom = max(1, int(len(window_means) * bottom_percent))
            if num_bottom == 1:
                return min(window_means)
            else:
                bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
                return np.mean(bottom_means)
        
        return 0.0
    except Exception:
        return 0.0


def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1) -> List[Dict[str, Any]]:
    """Filter traces by top confidence percentage"""
    if not traces:
        return []
    
    # Calculate confidences
    confidences = []
    for trace in traces:
        if confidence_type == 'mean':
            conf = calculate_mean_confidence(trace)
        elif confidence_type == 'tail':
            conf = calculate_tail_confidence(trace)
        elif confidence_type == 'bottom_window':
            conf = calculate_bottom_window_confidence(trace)
        elif confidence_type == 'min_window':
            conf = calculate_bottom_window_confidence(trace, bottom_percent=-1)
        else:
            conf = calculate_mean_confidence(trace)  # default fallback
        confidences.append(conf)
    
    # Get threshold for top percentage
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    
    # Filter traces
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if conf >= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces


def compute_all_voting_results(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute results for all voting methods
    
    Each trace has ["confs"] as per token confidences.
    And ["judge_confs"] as confidences of the same tokens from the judge model.
    """
    # Extract valid traces with answers
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]
    
    if not valid_traces:
        return {method: None for method in [
            'majority', 'mean_confidence_weighted', 'tail_confidence_weighted',
            'bottom_window_weighted', 'min_window_weighted', 
            'top10_tail_filtered', 'top10_bottom_window_filtered'
        ]}
    
    # Extract answers for voting
    answers = [trace['extracted_answer'] for trace in valid_traces]
    # Composite confidence aggregation:
    # - If a judge_conf array exists for a trace, combine the generator tail confidence
    #   and judge tail confidence via geometric mean (rewards agreement, penalizes disagreement).
    # - Otherwise, fall back to the generator tail confidence.
    def _composite_conf(trace: Dict[str, Any]) -> float:
        gen_tail = calculate_tail_confidence(trace)
        judge_confs = trace.get('judge_confs')
        if judge_confs:
            judge_tail = calculate_tail_confidence_from_confs(judge_confs)
            # geometric mean for non-negative confidences
            try:
                return float(np.sqrt(max(gen_tail, 0.0) * max(judge_tail, 0.0)))
            except Exception:
                return float(gen_tail)
        return float(gen_tail)

    # Prepare voting_results container early to avoid local-variable access issues
    voting_results = {}

    composite_confidences = [_composite_conf(t) for t in valid_traces]
    
    # Add composite-weighted vote (uses combined judge+generator signal when available)
    if any(c > 0 for c in composite_confidences):
        composite_answer = weighted_majority_vote(answers, composite_confidences)
        voting_results['composite_confidence_weighted'] = {
            'answer': composite_answer,
            'num_votes': len(answers),
            'confidence': float(np.mean(composite_confidences))
        }
    
    def calculate_confidence_variance(confs: List[float]) -> float:
        """Calculate variance of confidence scores"""
        if not confs or len(confs) < 2:
            return float('inf')  # High variance = low confidence
        return float(np.var(confs))

    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        gen_variances = [calculate_confidence_variance(t.get('confs', [])) for t in valid_traces]
        judge_variances = [calculate_confidence_variance(t.get('judge_confs', [])) for t in valid_traces]
        
        # Inverse variance weighting (lower variance = higher weight)
        dual_stability_weights = [
            1.0 / (1.0 + gv + jv) for gv, jv in zip(gen_variances, judge_variances)
        ]
        
        if any(w > 0 for w in dual_stability_weights):
            stability_answer = weighted_majority_vote(answers, dual_stability_weights)
            voting_results['dual_stability_weighted'] = {
                'answer': stability_answer,
                'num_votes': len(answers),
                'confidence': float(np.mean(dual_stability_weights))
            }

    def calculate_confidence_correlation(gen_confs: List[float], judge_confs: List[float]) -> float:
        """Calculate Pearson correlation between generator and judge confidences"""
        if not gen_confs or not judge_confs:
            return 0.0
        
        min_len = min(len(gen_confs), len(judge_confs))
        if min_len < 10:  # Need sufficient data
            return 0.0
        
        gen_arr = np.array(gen_confs[:min_len])
        judge_arr = np.array(judge_confs[:min_len])
        
        try:
            corr = np.corrcoef(gen_arr, judge_arr)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        correlations = [
            calculate_confidence_correlation(t.get('confs', []), t.get('judge_confs', []))
            for t in valid_traces
        ]
        
        # Filter top 20% by correlation
        top_corr_threshold = np.percentile(correlations, 80)
        high_corr_indices = [i for i, c in enumerate(correlations) if c >= top_corr_threshold]
        
        if high_corr_indices:
            high_corr_answers = [answers[i] for i in high_corr_indices]
            high_corr_weights = [tail_confidences[i] for i in high_corr_indices]
            
            corr_answer = weighted_majority_vote(high_corr_answers, high_corr_weights)
            voting_results['high_correlation_filtered'] = {
                'answer': corr_answer,
                'num_votes': len(high_corr_answers),
                'confidence': float(np.mean([correlations[i] for i in high_corr_indices]))
            }

        # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        def harmonic_mean(a: float, b: float) -> float:
            if a <= 0 or b <= 0:
                return 0.0
            return 2.0 * a * b / (a + b)
        
        dual_harmonic = [
            harmonic_mean(g, j) 
            for g, j in zip(tail_confidences, judge_tail_confidences)
        ]
        
        if any(w > 0 for w in dual_harmonic):
            harmonic_answer = weighted_majority_vote(answers, dual_harmonic)
            voting_results['dual_harmonic_tail_weighted'] = {
                'answer': harmonic_answer,
                'num_votes': len(answers),
                'confidence': float(np.mean(dual_harmonic))
            }

    def calculate_confidence_trend(confs: List[float], window_size: int = 512) -> float:
        """Calculate trend (slope) of confidence over time"""
        if not confs or len(confs) < window_size:
            return 0.0
        
        # Split into windows and calculate means
        num_windows = len(confs) // window_size
        if num_windows < 2:
            return 0.0
        
        window_means = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_means.append(np.mean(confs[start:end]))
        
        # Linear regression to find slope
        x = np.arange(len(window_means))
        try:
            slope = np.polyfit(x, window_means, 1)[0]
            return float(slope)
        except Exception:
            return 0.0

    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        gen_trends = [calculate_confidence_trend(t.get('confs', [])) for t in valid_traces]
        judge_trends = [calculate_confidence_trend(t.get('judge_confs', [])) for t in valid_traces]
        
        # Both models should show positive or neutral trend
        positive_trend_mask = [(gt >= 0 and jt >= 0) for gt, jt in zip(gen_trends, judge_trends)]
        
        trend_answers = [a for a, mask in zip(answers, positive_trend_mask) if mask]
        trend_weights = [tc for tc, mask in zip(tail_confidences, positive_trend_mask) if mask]
        
        if trend_answers and any(w > 0 for w in trend_weights):
            trend_answer = weighted_majority_vote(trend_answers, trend_weights)
            voting_results['positive_trend_filtered'] = {
                'answer': trend_answer,
                'num_votes': len(trend_answers),
                'confidence': float(np.mean(trend_weights)) if trend_weights else 0.0
            }

    def calculate_late_early_ratio(confs: List[float], split_point: float = 0.7) -> float:
        """Calculate ratio of late vs early confidence"""
        if not confs or len(confs) < 100:
            return 1.0
        
        split_idx = int(len(confs) * split_point)
        early_conf = np.mean(confs[:split_idx])
        late_conf = np.mean(confs[split_idx:])
        
        if early_conf <= 0:
            return 1.0
        
        return float(late_conf / early_conf)

    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        gen_ratios = [calculate_late_early_ratio(t.get('confs', [])) for t in valid_traces]
        judge_ratios = [calculate_late_early_ratio(t.get('judge_confs', [])) for t in valid_traces]
        
        # Both should show improvement (ratio > 1.0)
        improving_weights = [
            tail_confidences[i] * min(gr, jr)  # Weight by improvement
            for i, (gr, jr) in enumerate(zip(gen_ratios, judge_ratios))
            if gr >= 1.0 and jr >= 1.0
        ]
        improving_answers = [
            answers[i] for i, (gr, jr) in enumerate(zip(gen_ratios, judge_ratios))
            if gr >= 1.0 and jr >= 1.0
        ]
        
        if improving_answers and any(w > 0 for w in improving_weights):
            improving_answer = weighted_majority_vote(improving_answers, improving_weights)
            voting_results['improving_confidence_filtered'] = {
                'answer': improving_answer,
                'num_votes': len(improving_answers),
                'confidence': float(np.mean(improving_weights)) if improving_weights else 0.0
            }
    
    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        # Collect multiple confidence metrics
        metrics = {
            'tail': tail_confidences,
            'judge_tail': judge_tail_confidences,
            'geo_mean': [np.sqrt(g * j) for g, j in zip(tail_confidences, judge_tail_confidences)],
            'min': [min(g, j) for g, j in zip(tail_confidences, judge_tail_confidences)],
        }
        
        # Rank answers by each metric
        answer_scores = defaultdict(float)
        for metric_name, metric_values in metrics.items():
            # Sort indices by metric value (descending)
            ranked_indices = np.argsort(metric_values)[::-1]
            
            # Assign Borda scores (n-rank)
            n = len(ranked_indices)
            for rank, idx in enumerate(ranked_indices):
                answer = answers[idx]
                answer_scores[answer] += (n - rank)
        
        if answer_scores:
            borda_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x])
            voting_results['borda_count_multi_metric'] = {
                'answer': borda_answer,
                'num_votes': len(answers),
                'confidence': float(answer_scores[borda_answer] / (len(metrics) * len(answers)))
            }
        
    # В compute_all_voting_results:
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        trace_lengths = [len(t.get('confs', [])) for t in valid_traces]
        length_penalty = [1.0 / np.log(1 + length / 1000.0) for length in trace_lengths]
        
        dual_length_adjusted = [
            np.sqrt(g * j) * penalty
            for g, j, penalty in zip(tail_confidences, judge_tail_confidences, length_penalty)
        ]
        
        if any(w > 0 for w in dual_length_adjusted):
            length_adj_answer = weighted_majority_vote(answers, dual_length_adjusted)
            voting_results['dual_length_adjusted'] = {
                'answer': length_adj_answer,
                'num_votes': len(answers),
                'confidence': float(np.mean(dual_length_adjusted))
            }


    # Calculate different types of confidences (generator model)
    mean_confidences = [calculate_mean_confidence(trace) for trace in valid_traces]
    tail_confidences = [calculate_tail_confidence(trace) for trace in valid_traces]
    bottom_window_confidences = [calculate_bottom_window_confidence(trace) for trace in valid_traces]
    min_window_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in valid_traces]
    
    # 1. Simple majority vote
    majority_answer = simple_majority_vote(answers)
    voting_results['majority'] = {
        'answer': majority_answer,
        'num_votes': len(answers),
        'confidence': None
    }
    
    # 2. Mean confidence weighted vote
    if any(c > 0 for c in mean_confidences):
        mean_weighted_answer = weighted_majority_vote(answers, mean_confidences)
        voting_results['mean_confidence_weighted'] = {
            'answer': mean_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(mean_confidences)
        }
    
    # 3. Tail confidence weighted vote
    if any(c > 0 for c in tail_confidences):
        tail_weighted_answer = weighted_majority_vote(answers, tail_confidences)
        voting_results['tail_confidence_weighted'] = {
            'answer': tail_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(tail_confidences)
        }
    
    # 4. Bottom window confidence weighted vote
    if any(c > 0 for c in bottom_window_confidences):
        bottom_weighted_answer = weighted_majority_vote(answers, bottom_window_confidences)
        voting_results['bottom_window_weighted'] = {
            'answer': bottom_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(bottom_window_confidences)
        }
    
    # 5. Min window confidence weighted vote
    if any(c > 0 for c in min_window_confidences):
        min_window_answer = weighted_majority_vote(answers, min_window_confidences)
        voting_results['min_window_weighted'] = {
            'answer': min_window_answer,
            'num_votes': len(answers),
            'confidence': np.mean(min_window_confidences)
        }
    
    # 6. Top 10% tail confidence filtered + weighted vote
    top_tail_traces = filter_top_confidence(valid_traces, 'tail', 0.1)
    if top_tail_traces:
        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]
        
        if any(c > 0 for c in top_tail_confidences):
            top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
            voting_results['top10_tail_filtered'] = {
                'answer': top_tail_answer,
                'num_votes': len(top_tail_answers),
                'confidence': np.mean(top_tail_confidences)
            }
    
    # 7. Top 10% bottom window confidence filtered + weighted vote
    top_bottom_traces = filter_top_confidence(valid_traces, 'bottom_window', 0.1)
    if top_bottom_traces:
        top_bottom_answers = [trace['extracted_answer'] for trace in top_bottom_traces]
        top_bottom_confidences = [calculate_bottom_window_confidence(trace) for trace in top_bottom_traces]
        
        if any(c > 0 for c in top_bottom_confidences):
            top_bottom_answer = weighted_majority_vote(top_bottom_answers, top_bottom_confidences)
            voting_results['top10_bottom_window_filtered'] = {
                'answer': top_bottom_answer,
                'num_votes': len(top_bottom_answers),
                'confidence': np.mean(top_bottom_confidences)
            }
    
    # 8+. Dual-model metrics (if judge confidences are present)
    if any(('judge_confs' in t and t['judge_confs']) for t in valid_traces):
        judge_tail_confidences = [
            calculate_tail_confidence_from_confs(t.get('judge_confs', [])) for t in valid_traces
        ]

        # Min tail confidence (penalize disagreement)
        dual_min = [min(g, j) for g, j in zip(tail_confidences, judge_tail_confidences)]
        if any(w > 0 for w in dual_min):
            dual_min_answer = weighted_majority_vote(answers, dual_min)
            voting_results['dual_min_tail_weighted'] = {
                'answer': dual_min_answer,
                'num_votes': len(answers),
                'confidence': float(np.mean(dual_min))
            }

        # Geometric mean of tails (reward agreement)
        dual_geo = [np.sqrt(max(g, 0.0) * max(j, 0.0)) for g, j in zip(tail_confidences, judge_tail_confidences)]
        if any(w > 0 for w in dual_geo):
            dual_geo_answer = weighted_majority_vote(answers, dual_geo)
            voting_results['dual_geomean_tail_weighted'] = {
                'answer': dual_geo_answer,
                'num_votes': len(answers),
                'confidence': float(np.mean(dual_geo))
            }

        # Dual top-10% intersection by tail confidence
        try:
            thr_g = np.percentile(tail_confidences, 90)
            thr_j = np.percentile(judge_tail_confidences, 90)
        except Exception:
            thr_g, thr_j = 0.0, 0.0
        keep_pairs = [
            (a, np.sqrt(max(g, 0.0) * max(j, 0.0)))
            for a, g, j in zip(answers, tail_confidences, judge_tail_confidences)
            if g >= thr_g and j >= thr_j
        ]
        if keep_pairs:
            k_answers, k_weights = zip(*keep_pairs)
            dual_top_answer = weighted_majority_vote(list(k_answers), list(k_weights))
            voting_results['dual_top10_tail_filtered'] = {
                'answer': dual_top_answer,
                'num_votes': len(k_answers),
                'confidence': float(np.mean(list(k_weights)))
            }

    return voting_results


# ============= OUTPUT PROCESSING =============

def process_output(output, window_size: int) -> Dict[str, Any]:
    """Process a single vLLM output - for online mode with sliding window confidence"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    sliding_window = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store individual token confidences
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
    }


def process_batch_results(batch_outputs, window_size: int) -> Dict[str, Any]:
    """Process batch results from vLLM for a single question"""
    #print('Osize: ', len(batch_outputs))
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs
    # question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    min_confs = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output(output, window_size)
        traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'min_confs': min_confs,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }


def process_output_offline(output, window_size: int) -> Dict[str, Any]:
    """Process a single vLLM output for offline mode - stores full confidence array"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store full logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store full confidence array for offline analysis
        "extracted_answer": extracted_answer,
    }


def process_batch_results_offline(batch_outputs, window_size: int) -> Dict[str, Any]:
    """Process batch results from vLLM for offline mode"""
    # question_outputs = batch_outputs[0].outputs
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs

    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_offline(output, window_size)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }


