# Setup

- Pull repo
- Recreate env from uv.lock
- Download dataset via to_json_aime_2025.py

# HOW TO RUN

python offline_inference.py --model Qwen/Qwen3-4B-Instruct-2507-FP8 --judge Qwen/Qwen3-4B-Thinking-2507-FP8 --dataset aime_2025.jsonl --budget 256 --qid 0 3 --output_dir experiments/default/

You don't need to change other parameters, but if you want: 
Read --help. It's pretty self explanatory.
If vLLM yells at you for not having enough memory -> reduce --max_model_len
To speed up testing -> reduce --budget and --max_tokens

# offline_inference logic:

- Load question -> prompt
- Load model (does the CoT generation)
- Load judge (just computes confidence for generated CoT)
- deepthink
    - _deepthink_offline
        - batched generation of CoT traces
        - process_output_offline -> traces, confs
    - _score_traces_with_judge
    - compute_multiple_voting
- evaluate_voting_results: check if voted_answer == ground_truth

# YOUR CODE:
deepconf/deepconf/utils.py

You basically need to use per token conf from generation model and from judge to filter out bad CoT tracs.

# What experiments to run?

You will write functions to aggregate confidences.
Reinstall the package
uv pip uninstall deepconf
uv pip install ./deepconf
And run offline_inference.py with different models

First test out with small budget and a couple of questions.
After completion of each question it will print out your confidence aggregation performance.


Then do with BIDGET = 256 and don't specify qid. It will inference the whole dataset (the output directory for this final run should be empty befor starting!!!)

Run aggregate_statistics.py to calculate performance acros the whole dataset.

Baseline: Some confidence metrics that don't use judge are already implemented and calculated. Have you beaten them? YOU ARE COOL!!! (and deserve a publication)

# Models to try out
- [ ] Qwen/Qwen3-4B-Thinking-2507
- [ ] Qwen/Qwen3-4B-Thinking-2507-FP8
- [ ] Qwen/Qwen3-4B-Instruct-2507 ~47%
- [ ] Qwen/Qwen3-4B
- [ ] Qwen/Qwen3-Next-80B-A3B-Instruct
- [ ] Qwen/Qwen3-30B-A3B-Instruct-2507
- [ ] Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
- [ ] deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- Everything from the Qwen3 family (or basically anything that has Qwen3 tokenizer)

# Possible experiments:
first read through paper's already implemented methods.
For example top10_bottom_window_filtered is the best baseline conf metric.

- Aggregation method ideas: 
    - Avg(conf, judge_conf)
        - [x] dual_geomean_tail_weighted
    - Min (conf, judge_conf)
        - [x] min(tail_conf, tail_judge_conf) - already implemented
        - [x] dual_top10_tail_filtered
    - Consensus - trace is considered only if it is in top10 percentile by conf and judge_conf (both favor it)

- Model generator/judge combinations:
    - Thinking/instruct/base (I have a suspicion that thinking/non-thinking models will disagree constantly. So we will need to test inside respective thinking class)
    - Full/FP8
    - 32B/4B (I recon that big generator with small judge would be good)
    - Next(disable NTP or set NTP=1)/others/coder/VL? Use everything based on qwen3 tokenizer. Go wild.

