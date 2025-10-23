from offline_inference import clean_thinking_output, prepare_prompt, prepare_prompt_gpt
from deepconf.wrapper import DeepThinkLLM
from vllm import SamplingParams
import argparse
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='DeepThinkLLM Offline Mode Example')
    parser.add_argument('--model', type=str, default="openai/gpt-oss-120b",
                       help='Model path or name')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for model')
    parser.add_argument('--dataset', type=str, default="aime_2025.jsonl",
                       help='Dataset file path')
    parser.add_argument('--qid', type=int, nargs='*',
                        help='List of question IDs to process (0-based indices). If not provided, will process the whole dataset')
    parser.add_argument('--rid', type=str, default="offline_run",
                       help='Run ID for identification')
    parser.add_argument('--budget', type=int, default=256,
                       help='Number of traces to generate')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Sliding window size for confidence computation')
    parser.add_argument('--max_tokens', type=int, default=10000,
                       help='Maximum tokens per generation')
    parser.add_argument('--max_model_len', type=int, default=32768,
                       help='Maximum model length for KV caching')
    parser.add_argument('--model_type', type=str, default="gpt", choices=["deepseek", "gpt","qwen_thinking", "qwen_next"],
                       help='Model type for prompt formatting')
    parser.add_argument('--reasoning_effort', type=str, default="high",
                       help='Reasoning effort for GPT models')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Nucleus sampling top-p value')
    parser.add_argument('--top_k', type=int, default=-1,
                       help='Top-k sampling value')
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    parser.add_argument("--gpu_memory_utilization", type=float,default=0.95,
                       help="Desired GPU memory utilization ratio (0 to 1) for model loading")
    parser.add_argument("--max_num_batched_tokens", type=int,default=16384,
                        help='Number of tokens that the model can process in one batch')
    parser.add_argument("--max_num_seqs", type=int,default=512,
                        help='Number of requests in one time')
    parser.add_argument("--chunked_prefill", action='store_true',
                       help='Enable vLLM chunked prefill scheduling policy')
    parser.add_argument("--gpu", type=int, default=0,
                       help='GPU ID')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # run_name now summarizes key model and generation-related parameters for easy identification
    args.run_name = (
        f"model{args.model.split('/')[-1]}_"
        f"T{args.temperature}_"
        f"max_tok{args.max_tokens}_"
        f"max_model_len{args.max_model_len}_"
    )
    
    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
    
    assert args.temperature != 0, "Temperature must not be 0"
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    
    init_kwargs = {
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'enable_prefix_caching': True,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_num_seqs': args.max_num_seqs,
        'enable_chunked_prefill': args.chunked_prefill,
    }

    # Throughput optimization: allow all n-completions to decode concurrently when possible
    if args.max_num_seqs < args.budget:
        print(f"max_num_seqs ({args.max_num_seqs}) < budget ({args.budget}); setting max_num_seqs to {args.budget} for better concurrency")
        init_kwargs['max_num_seqs'] = args.budget
    
    deep_llm = DeepThinkLLM(**init_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )
    context = []
    
    qids = args.qid if args.qid else range(len(data))
    
    for qid in tqdm(qids, desc="Processing questions"):
        question_data = data[qid]
        question = question_data['question']
        ground_truth = str(question_data.get('answer', '')).strip()
        if args.model_type == "gpt":
            prompt = prepare_prompt_gpt(question, deep_llm.tokenizer, args.reasoning_effort)
        else:
            prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)
        context.append((prompt, ground_truth, question, qid))
    
    for prompt, ground_truth, question, qid in tqdm(context, desc="Running inference"):
        # Run deep thinking in offline mode
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="offline",
            budget=args.budget,
            window_size=args.window_size,
            sampling_params=sampling_params,
        )
        
        # Clean thinking blocks from outputs if using Qwen Thinking
        if args.model_type == "qwen_thinking":
            for trace in result.all_traces:
                if 'text' in trace:
                    trace['text'] = clean_thinking_output(trace['text'])
                if 'extracted_answer' in trace:
                    trace['extracted_answer'] = clean_thinking_output(trace['extracted_answer'])

        
        # Save results
        
        os.makedirs(os.path.join(args.output_dir, args.run_name, f"qid{qid}"), exist_ok=True)
        
        with open(os.path.join(args.output_dir, args.run_name, f"qid{qid}", "traces.json"), 'w') as f:
            json.dump(result.all_traces, f)
        
        # Count stop reasons
        stop_reasons = {}
        for trace in result.all_traces:
            reason = trace.get('stop_reason', 'unknown')
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        
        # Print statistics
        total_traces = len(result.all_traces)
        length_stopped = stop_reasons.get('length', 0)
        
        print(f"\n{'='*60}")
        print(f"Stop Reason Statistics for qid{qid}:")
        print(f"  Total traces: {total_traces}")
        print(f"  Stopped due to length: {length_stopped} ({100*length_stopped/total_traces:.1f}%)")
        for reason, count in sorted(stop_reasons.items()):
            if reason != 'length':
                print(f"  Stopped due to '{reason}': {count} ({100*count/total_traces:.1f}%)")
        print(f"{'='*60}")
        
        print(f"\nResults saved to {os.path.join(args.output_dir, args.run_name, f"qid{qid}", "traces.json")}")


if __name__ == "__main__":
    main()