"""
DeepThinkLLM implementation with online and offline mode support

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import numpy as np
from typing import Optional, Dict, Any
import os
import copy

from .outputs import DeepThinkOutput
from .utils import (
    process_batch_results, process_batch_results_offline,
    weighted_majority_vote, compute_all_voting_results, compute_confidence
)
from .processors import WrappedPerReqLogitsProcessor


class DeepThinkLLM:
    """Enhanced LLM wrapper with deep thinking capabilities"""
    
    def __init__(self, model: str, judge_model: Optional[str] = None, **vllm_kwargs):
        """
        Initialize DeepThinkLLM with optimized defaults for batching
        
        Args:
            model: Model path or name
            judge_model: Optional judge model path or name for dual-confidence scoring (offline only)
            **vllm_kwargs: Additional arguments for vLLM initialization
        """
        self.model_name = model
        self.judge_model_name = judge_model
        self.vllm_kwargs = vllm_kwargs
        
        # Initialize vLLM with optimized defaults for batching performance
        default_kwargs = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            # Batching optimizations
            "max_num_batched_tokens": 32768,  # Larger batch size for better throughput
            "max_num_seqs": 256,  # Support more sequences in batch
        }
        default_kwargs.update(vllm_kwargs)
        
        print("Initializing vLLM engine with optimized batching...")
        print(f"  - Prefix caching: {default_kwargs.get('enable_prefix_caching', False)}")
        print(f"  - Max batched tokens: {default_kwargs.get('max_num_batched_tokens', 'default')}")
        print(f"  - Max sequences: {default_kwargs.get('max_num_seqs', 'default')}")
        llm_init_start = time.time()
        self.llm = LLM(model=model, logits_processors=[WrappedPerReqLogitsProcessor], **default_kwargs)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        # Optional judge LLM (uses same tokenizer assumption)
        self.judge_llm = None
        if judge_model:
            if judge_model == model:
                print("Judge model equals generator; reusing LLM engine for judging.")
                self.judge_llm = self.llm
            else:
                print("Initializing judge vLLM engine...")
                judge_defaults = {
                    "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
                    "enable_prefix_caching": True,
                    "trust_remote_code": True,
                    "max_num_batched_tokens": 32768,
                    "max_num_seqs": 256,
                }
                judge_defaults.update({k: v for k, v in vllm_kwargs.items() if k in (
                    "tensor_parallel_size", "enable_prefix_caching", "trust_remote_code",
                    "max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization"
                )})
                # Constrain GPU memory utilization for judge if not provided
                judge_defaults.setdefault("gpu_memory_utilization", 0.6)
                judge_init_start = time.time()
                try:
                    self.judge_llm = LLM(model=judge_model, logits_processors=[WrappedPerReqLogitsProcessor], **judge_defaults)
                    print(f"Judge vLLM engine initialized in {time.time() - judge_init_start:.2f} seconds")
                except Exception as e:
                    print(f"Failed to initialize judge LLM due to: {e}. Disabling judge scoring.")
                    self.judge_llm = None

        # Store initialization times
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
    
    def generate(self, *args, **kwargs):
        """Simple wrapper around vLLM's generate method"""
        return self.llm.generate(*args, **kwargs)
    
    def deepthink(
        self,
        prompt: str,
        mode: str = "offline",
        # Online mode parameters
        warmup_traces: int = 16,
        total_budget: int = 256,
        confidence_percentile: int = 90,
        # Offline mode parameters  
        budget: int = 512,
        # Common parameters
        window_size: int = 2048,
        sampling_params: Optional[SamplingParams] = None,
        # Multiple voting options
        compute_multiple_voting: bool = True,
        **kwargs
    ) -> DeepThinkOutput:
        """
        Perform deep thinking on a prompt
        
        Args:
            prompt: Input prompt (prepared string)
            mode: "online" for confidence-based early stopping, "offline" for batch generation
            warmup_traces: Number of warmup traces for online mode
            total_budget: Total budget for online mode
            confidence_percentile: Percentile for confidence threshold in online mode
            budget: Number of traces for offline mode
            window_size: Window size for confidence computation
            sampling_params: Custom vLLM sampling parameters
            compute_multiple_voting: Whether to compute multiple voting method results
            
        Returns:
            DeepThinkOutput containing results
        """
        total_start_time = time.time()
        
        # Create output object
        output = DeepThinkOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=32000,
                logprobs=20,
            )
            
        # Set configuration
        output.config = {
            "model": self.model_name,
            "mode": mode,
            "window_size": window_size,
            "compute_multiple_voting": compute_multiple_voting,
        }
        
        if mode == "online":
            output.config.update({
                "warmup_traces": warmup_traces,
                "total_budget": total_budget,
                "confidence_percentile": confidence_percentile,
            })
            result = self._deepthink_online(
                prompt, output, 
                warmup_traces, total_budget, confidence_percentile,
                window_size, sampling_params
            )
        else:
            output.config.update({
                "budget": budget,
            })
            result = self._deepthink_offline(
                prompt, output,
                budget, window_size, sampling_params
            )
        
        # Offline-only: if judge is available, score traces before voting
        if self.judge_llm and result.all_traces and output.mode == "offline":
            self._score_traces_with_judge(self.judge_llm, prompt, output.all_traces, window_size)
            output.config["judge_model"] = self.judge_model_name

        # Perform multiple voting analysis if requested
        if compute_multiple_voting and output.all_traces:
            print("Computing multiple voting results...")
            voting_start = time.time()
            if output.mode == "online":
                output.voting_results = compute_all_voting_results(output.all_voting_traces)
            else:   
                output.voting_results = compute_all_voting_results(output.all_traces)
            
            # Set the primary answer to the majority vote result
            if 'majority' in output.voting_results and output.voting_results['majority']:
                output.voted_answer = output.voting_results['majority']['answer']
                output.final_answer = output.voted_answer
            
            voting_time = time.time() - voting_start
            print(f"Multiple voting computed in {voting_time:.2f} seconds")
        
        output.total_time = time.time() - total_start_time
        output.print_summary()
        
        if compute_multiple_voting and output.voting_results:
            output.print_detailed_voting_results()
        
        return output

    def _score_traces_with_judge(self, judge_llm: LLM, prompt: str, traces, window_size: int) -> None:
        """Score generated continuations with a judge model using prefill logprobs.

        Assumes the same tokenizer is used for counting token boundaries.
        Stores per-token confidences under trace['judge_confs'].
        """
        if not traces:
            return

        # Build combined prompts and measure continuation lengths in tokens
        combined_prompts = []
        continuation_token_lengths = []

        # Compute prompt tokenization once
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids

        for trace in traces:
            generated_text = trace.get("text", "")
            combined_prompts.append(prompt + generated_text)
            cont_ids = self.tokenizer(generated_text, add_special_tokens=False).input_ids
            continuation_token_lengths.append(len(cont_ids))

        # Prefill scoring: no generation, return prompt logprobs
        judge_params = SamplingParams(max_tokens=0, prompt_logprobs=1)

        judge_outputs = judge_llm.generate(combined_prompts, judge_params)

        # Extract prompt logprobs and slice to continuation region
        for idx, req_out in enumerate(judge_outputs):
            cont_len = continuation_token_lengths[idx]
            if cont_len == 0:
                traces[idx]["judge_confs"] = []
                continue

            # vLLM returns prompt_logprobs as a list aligned with prompt tokens
            plps = getattr(req_out, "prompt_logprobs", None)
            # Some versions may place it under outputs[0]
            if plps is None and getattr(req_out, "outputs", None):
                first_out = req_out.outputs[0]
                plps = getattr(first_out, "prompt_logprobs", None)

            if not plps:
                traces[idx]["judge_confs"] = []
                continue

            # Slice last cont_len positions to target continuation tokens
            cont_plps = plps[-cont_len:]
            try:
                judge_confs = compute_confidence(cont_plps) if cont_plps else []
            except Exception:
                judge_confs = []

            traces[idx]["judge_confs"] = judge_confs
    
    def _deepthink_online(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        total_budget: int,
        confidence_percentile: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """Online deep thinking with confidence-based early stopping"""
        
        processing_start = time.time()
        
        # Warmup phase
        print(f"Starting warmup phase...", sampling_params)
        warmup_gen_start = time.time()
        

        # Generate warmup traces
        warmup_params_list = []
        base_seed = time.time_ns()
        for param_id in range(warmup_traces):
            warmup_params = copy.deepcopy(sampling_params) 
            warmup_params.logprobs = 20
            warmup_params.seed = base_seed + param_id
            warmup_params_list.append(warmup_params)
        warmup_outputs = self.llm.generate([prompt for _ in range(warmup_traces)], warmup_params_list)
        output.warmup_gen_time = time.time() - warmup_gen_start
        
        # Process warmup results
        warmup_process_start = time.time()
        warmup_result = process_batch_results(warmup_outputs, window_size)
        output.warmup_process_time = time.time() - warmup_process_start
        
        print('Warmup min_confs:', warmup_result['min_confs'])
        output.conf_bar = float(np.percentile(warmup_result['min_confs'],100 - confidence_percentile))
        output.warmup_min_confs = warmup_result['min_confs']
        
        output.warmup_traces = warmup_result['traces']
        output.warmup_tokens = warmup_result['total_tokens']
        
        print(f"Warmup completed: conf_bar={output.conf_bar:.3f}")
        
        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()
        # final_params = copy.deepcopy(sampling_params)
        # final_params.seed = int(time.time())
        # final_params.n = total_budget - warmup_traces

        final_params_list = []
        for param_id in range(total_budget - warmup_traces):
            final_params = copy.deepcopy(sampling_params) 
            final_params.logprobs = 20
            final_params.seed = base_seed + param_id + warmup_traces
            final_params.extra_args = {
                "conf_threshold": output.conf_bar,
                "eos_token_id": self.tokenizer.eos_token_id,
                "conf_group_size": window_size,
                "conf_topk": 20,
            }
            final_params_list.append(final_params)
        final_outputs = self.llm.generate([prompt for _ in range(total_budget - warmup_traces)], final_params_list)
        output.final_gen_time = time.time() - final_gen_start
        
        # Process final results
        final_process_start = time.time()
        final_result = process_batch_results(final_outputs, window_size)
        output.final_process_time = time.time() - final_process_start
        
        print('Final min_confs:', final_result['min_confs'])
        output.final_min_confs = final_result['min_confs']
        
        output.final_traces = final_result['traces']
        output.final_tokens = final_result['total_tokens']
        
        # Apply confidence threshold to final traces
        for trace in output.final_traces:
            if trace["min_conf"] < output.conf_bar:
                trace["stop_reason"] = "gconf_threshold"
        
        # Combine all traces
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def _deepthink_offline(
        self,
        prompt: str,
        output: DeepThinkOutput,
        budget: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
        """
        Offline deep thinking - generate all traces at once using optimized batching.
        
        OPTIMIZATION: Uses vLLM's native 'n' parameter with a single prompt instead of 
        creating budget duplicate prompts. This enables true batched inference with 
        prefix caching, resulting in 5-10x speedup.
        """
        
        # OPTIMIZED: Use native n parameter instead of duplicate prompts
        # This allows vLLM to share the KV cache for the prompt across all completions
        print(f"Generating {budget} traces with optimized batching...")
        generation_start = time.time()
        
        # Create sampling params with n=budget for true batched generation
        optimized_params = copy.deepcopy(sampling_params)
        optimized_params.n = budget
        optimized_params.logprobs = 20
        optimized_params.use_beam_search = False  # Ensure sampling mode
        
        # Set base seed for reproducibility (each completion gets seed + offset automatically)
        if not hasattr(optimized_params, 'seed') or optimized_params.seed is None:
            optimized_params.seed = int(time.time_ns() % (2**31))
        
        # Single prompt with n completions (much faster than n duplicate prompts)
        vllm_outputs = self.llm.generate([prompt], [optimized_params])
        output.generation_time = time.time() - generation_start
        
        print(f"Generation completed in {output.generation_time:.2f}s")
        
        # Process results
        processing_start = time.time()
        processed_results = process_batch_results_offline(vllm_outputs, window_size)
        
        output.all_traces = processed_results['traces']
        output.total_tokens = processed_results['total_tokens']
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        print(f"Processing completed in {output.processing_time:.2f}s")
        
        return output
    
    def _perform_basic_voting(self, output: DeepThinkOutput):
        """Perform basic weighted majority voting (for backward compatibility)"""
        voting_answers = []
        voting_weights = []
        
        if output.mode == "online":
            output.all_voting_traces = []
            # Add warmup traces above threshold
            for trace in output.warmup_traces:
                if trace.get('min_conf', 0) >= output.conf_bar and trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
                    output.all_voting_traces.append(trace)
            
            # Add final traces (skip early stopped ones)
            for trace in output.final_traces:
                if trace.get('stop_reason') == 'gconf_threshold':
                    continue
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(trace.get('min_conf', 1.0))
                    output.all_voting_traces.append(trace)
        else:
            # Offline mode - use all traces with valid answers
            for trace in output.all_traces:
                if trace.get('extracted_answer'):
                    voting_answers.append(trace['extracted_answer'])
                    voting_weights.append(1.0)
        
        output.voting_answers = voting_answers
        output.voting_weights = voting_weights
        
        # Get voted answer (basic method)
        output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        output.final_answer = output.voted_answer
        
        # Calculate token statistics
        if output.mode == "online":
            output.avg_tokens_per_warmup_trace = output.warmup_tokens / len(output.warmup_traces) if output.warmup_traces else 0
            output.avg_tokens_per_final_trace = output.final_tokens / len(output.final_traces) if output.final_traces else 0
        
        print(f'Basic voting candidates: {len(voting_answers)}')
        if voting_answers:
            print(f'Sample voting answers: {voting_answers[:5]}')