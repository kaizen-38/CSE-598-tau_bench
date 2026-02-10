# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains

**❗News**: We have released [τ²-bench](https://github.com/sierra-research/tau2-bench) as an extension of $\tau$-bench. $\tau^2$-bench includes code fixes and an additional `telecom` domain focusing on troubleshooting scenarios. Please use the $\tau^2$-bench as the latest version of this benchmark.

**Paper**:
* [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
* [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982)

We propose $\tau$-bench, a benchmark emulating dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines.

## Leaderboard

### Airline

| Strategy       | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
| -------------- | ------ | ------ | ------ | ------ |
| [TC (claude-3-5-sonnet-20241022)](https://www.anthropic.com/news/3-5-models-and-computer-use)      | **0.460**     | **0.326**     | **0.263**     | **0.225**     |
| [TC (gpt-4o)](https://platform.openai.com/docs/guides/function-calling)     | 0.420     | 0.273     | 0.220     | 0.200     |
| [TC (claude-3-5-sonnet-20240620)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)      | 0.360     | 0.224     | 0.169     | 0.139     |
| [TC (mistral-large-2407)](https://docs.mistral.ai/capabilities/function_calling/)     | ??     | ??     | ??     | ??     |
| [TC (gpt-4o-mini)](https://platform.openai.com/docs/guides/function-calling)     | 0.225     | 0.140     | 0.110     | 0.100     |
| [Act](https://arxiv.org/abs/2210.03629) (gpt-4o)     | 0.365 | 0.217 | 0.160 | 0.140     |
| [ReAct](https://arxiv.org/abs/2210.03629) (gpt-4o)     | 0.325 | 0.233 | 0.185 | 0.160     |

### Retail

| Strategy       | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
| -------------- | ------ | ------ | ------ | ------ |
| [TC (claude-3-5-sonnet-20241022)](https://www.anthropic.com/news/3-5-models-and-computer-use)      | **0.692**     | **0.576**     | **0.509**     | **0.462**     |
| [TC (gpt-4o)](https://platform.openai.com/docs/guides/function-calling)     | 0.604     | 0.491     | 0.430     | 0.383     |
| [TC (claude-3-5-sonnet-20240620)](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)      | 0.626     | 0.506     | 0.435     | 0.387     |
| [TC (mistral-large-2407)](https://docs.mistral.ai/capabilities/function_calling/)     | ??     | ??     | ??     | ??     |
| [TC (gpt-4o-mini)](https://platform.openai.com/docs/guides/function-calling)     | ??     | ??     | ??     | ??     |
| [Act](https://arxiv.org/abs/2210.03629) (gpt-4o)     | ??     | ??     | ??     | ??     |
| [ReAct](https://arxiv.org/abs/2210.03629) (gpt-4o)     | ??     | ??     | ??     | ??     |

*TC = `tool-calling` strategy (the function-calling strategy reported in the paper)

## Setup

1. Clone this repository:

```bash
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

## Run

Run a tool-calling agent on the τ-retail environment:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10
```

Set max concurrency according to your API limit(s).

To run specific tasks, use the `--task-ids` flag. For example:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2 4 6
```

This command will run only the tasks with IDs 2, 4, and 6.

## User simulators

By default, we use `gpt-4o` as the user simulator with strategy `llm`. You can use other models by setting the `--user-model` flag, or other strategies by setting the `--user-strategy` flag. For example, run a tool-calling agent with a claude user simulator:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --max-concurrency 10 --user-model claude-3-5-sonnet-20240620 --user-model-provider anthropic --user-strategy llm
```

Other strategies:

To run `react` user simulator:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --max-concurrency 10 --user-model gpt-4o --user-model-provider openai --user-strategy react
```

Example of a `react` user response:

```md
Thought:
I should provide my name and zip code as I wasn't given an email address to use.

User Response:
Sure, my name is Yusuf Rossi, and my zip code is 19122.
```

To run `verify` user simulator:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --max-concurrency 10 --user-model gpt-4o --user-model-provider openai --user-strategy verify
```

This strategy uses a subsequent LLM verification step to check if the user simulator's response is satisfactory. If not, the user simulator will be prompted to generate a new response.

To run `reflection` user simulator:

```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --max-concurrency 10 --user-model gpt-4o --user-model-provider openai --user-strategy reflection
```

This strategy uses a subsequent LLM verification step to check if the user simulator's response is satisfactory. If not, the user simulator will be prompted to reflect on its response and generate a new response.

## Running on Intel Gaudi (ASU Sol HPC)

This fork adds support for running tau-bench on Intel Gaudi accelerators via SLURM, using Qwen3-32B as the agent model and a Voyager-hosted Qwen3-30B as the user simulator.

### Prerequisites

- Access to a SLURM cluster with Intel Gaudi nodes (e.g., ASU Sol)
- The vLLM-Gaudi container at `/data/sse/gaudi/containers/vllm-gaudi.sif`
- A Python virtual environment with `litellm` and tau-bench installed at `/scratch/$USER/tau-bench-venv`
- Voyager API keys for the user simulation model

### Initial Setup on the Cluster

```bash
# Clone and install
cd /scratch/$USER
git clone https://github.com/kaizen-38/CSE-598-tau_bench.git tau-bench
cd tau-bench

# Create virtual environment
python3 -m venv /scratch/$USER/tau-bench-venv
source /scratch/$USER/tau-bench-venv/bin/activate
pip install -e .
pip install litellm

# Create required directories
mkdir -p logs results
```

### Submitting Experiments

Each experiment is a self-contained SLURM script that launches a vLLM server on Gaudi, waits for readiness, then runs tau-bench.

```bash
cd /scratch/$USER/tau-bench

# --- Retail domain ---
sbatch run_gaudi_retail_react.sh          # ReAct strategy
sbatch run_gaudi_retail_act.sh            # Act strategy
sbatch run_gaudi_retail_toolcalling.sh    # Tool-calling strategy

# --- Airline domain ---
sbatch run_gaudi_airline_react.sh         # ReAct strategy
sbatch run_gaudi_airline_act.sh           # Act strategy
sbatch run_gaudi_airline_toolcalling.sh   # Tool-calling strategy
```

### Experiment Configuration

All scripts share the same defaults (edit the script to change):

| Parameter | Value |
|-----------|-------|
| Agent model | `Qwen/Qwen3-32B` (local vLLM on Gaudi) |
| User model | `qwen3-30b-a3b-instruct-2507` (Voyager API) |
| Temperature | 0.7 |
| Num trials | 5 (pass^1 through pass^5) |
| Concurrency | 6 |
| Max model length | 16384 |
| SLURM time limit | 24 hours |

**Note on tool-calling scripts:** The tool-calling strategy requires vLLM to be started with `--enable-auto-tool-choice --tool-call-parser hermes` (already included in the scripts). This is required for Qwen models to handle native tool calling via the OpenAI-compatible API.

### Monitoring Jobs

```bash
# List your running/pending jobs
squeue -u $USER

# Watch a job's live output (replace JOBID)
tail -f logs/tau-gaudi-retail-react-JOBID.out

# Check for errors
tail -f logs/tau-gaudi-retail-react-JOBID.err

# Cancel a job
scancel JOBID
```

### Results

Results are saved as JSON checkpoint files in `results/gaudi-{domain}-{strategy}/`. Each file contains an array of task results with fields:

- `task_id`: the benchmark task number
- `reward`: 1.0 for pass, 0.0 for fail
- `trial`: trial number (0-4 for 5 trials)
- `info`: metadata including error messages if the task failed
- `traj`: the full conversation trajectory

### Analyzing Results

Use `analyze_failures.py` to inspect results and verify failures are from the benchmark (not infrastructure issues):

```bash
# Full summary of all ACT & REACT experiments
python3 analyze_failures.py

# Filter by strategy or domain
python3 analyze_failures.py --strategy act
python3 analyze_failures.py --strategy react
python3 analyze_failures.py --domain airline
python3 analyze_failures.py --domain retail

# Deep-dive into a specific task across all trials
python3 analyze_failures.py --task-id 42

# Print task-by-task pass/fail log (like HPC terminal output)
python3 analyze_failures.py --log --domain retail --strategy act
python3 analyze_failures.py --log --domain retail --strategy act --trial 0

# Show only trial 0 (pass^1) for airline react
python3 analyze_failures.py --log --domain airline --strategy react --trial 0
```

Use `aggregate_results.py` to merge results from multiple runs and check completeness:

```bash
# Aggregate and check for missing tasks
python3 aggregate_results.py results/gaudi-airline-act \
    -o results/agg-airline-act.json \
    --check-missing --domain airline

python3 aggregate_results.py results/gaudi-retail-react \
    -o results/agg-retail-react.json \
    --check-missing --domain retail
```

### Running Specific Tasks

To re-run only specific task IDs (e.g., to retry infrastructure failures):

```bash
python run.py \
    --agent-strategy act \
    --env airline \
    --model Qwen/Qwen3-32B \
    --model-provider openai \
    --user-model qwen3-30b-a3b-instruct-2507 \
    --user-model-provider openai \
    --user-strategy llm \
    --temperature 0.7 \
    --num-trials 5 \
    --task-ids 0 13 22 \
    --log-dir results/gaudi-airline-act
```

---

## Auto error identification

Often times, it is difficult and time consuming to manually identify specific error locations in trajectories as they can be long and the constraints can be complex. We have provided an auto error identification tool that can do the following:

1. Fault assignment: determine the entity that is responsible for the fault (user, agent, environment)
2. Fault type classification: classify the type of fault (goal_partially_completed, used_wrong_tool, used_wrong_tool_argument, took_unintended_action)

Both of the labels are accompanied with a description.

To run the auto error identification, run:

```bash
python auto_error_identification.py --env <airline/retail> --platform openai --results-path <the path to your results file here> --max-concurrency 16 --output-path test-auto-error-identification --max-num-failed-results 10
```

Please note that this feature utilizes an LLM, which may lead to inaccurate error identifications.

*Notice: If an error is raised due to the structure of your results file, you may have to rerun the benchmark to produce a new results file. We have recently [rewritten](https://github.com/sierra-research/tau-bench/commit/043b544371757ebb3762b3d02a6675dfe0c41798) the benchmark to be more type-safe and extensible.

## Historical trajectories

τ-bench might be expensive to run. We have provided a set of historical trajectories for the airline and retail environments in `./historical_trajectories`.

If you would like to contribute your historical trajectories to this benchmark, please submit a PR!

## License

See `./LICENSE`.

## Contact

Please submit issues or pull requests if you find problems with the benchmark.

## Citation

```bibtex
@misc{yao2024tau,
      title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains}, 
      author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
      year={2024},
      eprint={2406.12045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.12045}, 
}
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```
