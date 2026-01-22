# SWE-Reflexion

SWE-Reflexion is an automated code bug localization framework driven by Large Language Model (LLM) APIs (e.g., GPT-4, DeepSeek). Unlike traditional one-way search methods, this project adopts a **Multi-Agent Collaboration** mechanism. Through the interaction of the main agent (Explorer), the critique agent (Critic), and the reflection agent (Reflector), combined with **Graph Index** and **BM25 Index** to enhance retrieval capabilities, it achieves high-precision bug localization.

## Core Features

* **Multi-Agent Collaboration**:
* **Explorer**: Executes code search, reading, and reasoning.
* **Critic**: Reviews the Explorer's path and points out invalid explorations.
* **Reflector**: Summarizes experience and adjusts search strategies when stuck.


* **Hybrid Retrieval Enhancement**:
* Integrates **Code Property Graph (CPG) Index** for structured navigation.
* Integrates **BM25 Sparse Index** for precise keyword matching.


* **Self-Reflection & Backtracking**: Supports self-correction based on interaction history to avoid wasting tokens on wrong paths.

## Prerequisites

### 1. Install Dependencies

Ensure Python version >= 3.9, then install the project dependencies:

```bash
pip install -r requirements.txt

```

### 2. Configure Environment Variables (Critical)

Create a `.env` file in the project root directory and configure the paths and keys strictly following the format below. This is essential for the system to run.

**Example `.env` file:**

```ini
# ==============================
# Model API Configuration (Required)
# ==============================
# OpenAI-compatible API endpoint address
OPENAI_API_BASE="https://api.deepseek.com/v1"
# Your API Key
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Specify the model name (Must match the name supported by the API provider)
MODEL_NAME="deepseek-chat"

# ==============================
# Index Data Path Configuration (Required)
# ==============================
# Graph Index Directory (Used for structured traversal explore_tree_structure)
GRAPH_INDEX_DIR="/path/to/your/project/index_data/graph_index"

# BM25 Sparse Index Directory (Used for keyword search keyword_search)
BM25_INDEX_DIR="/path/to/your/project/index_data/bm25_index"

# Vector Index Directory (Used for semantic search semantic_search)
VECTOR_INDEX_DIR="/path/to/your/project/index_data/vector_index"

# Repository Cache Directory (Used for git operations and reading source code)
REPO_CACHE_DIR="./playground/repo"

```

> **Tip**: Please ensure the above index directories contain pre-built index files for the target dataset (e.g., SWE-bench).

## Quick Start

### Run Localization Script

Use `auto_search_main_single.py` to start the localization task. The script will automatically read the configuration from `.env` to connect to the model and indexes.

```bash
python auto_search_main_single.py \
    --output_folder ./results_api_test \
    --num_samples 1 \
    --max_attempt_num 2 \
    --use_function_calling

```

### Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | (Read from .env) | Specifies the LLM model name (can be overridden via command line) |
| `--output_folder` | `lite_raw_res` | Results output directory |
| `--dataset` | `czlll/SWE-bench_Lite` | Dataset used for evaluation |
| `--eval_n_limit` | 274 | Limit the number of evaluation samples |
| `--critic_agent` | `True` | Enable Critic Agent (Recommended) |
| `--reflection_agent` | `True` | Enable Reflector Agent (Recommended) |
| `--use_function_calling` | `True` | Enable Function Calling format (Recommended) |
| `--enable_semantic_search` | `True` | Enable usage of semantic search tools |

### Running Examples

**1. Run full test using DeepSeek-V3:**

```bash
python auto_search_main_single.py \
    --model "deepseek-chat" \
    --output_folder "./results/deepseek_v3" \
    --use_function_calling \
    --critic_agent True

```

**2. Run only the first 10 samples for quick verification:**

```bash
python auto_search_main_single.py \
    --eval_n_limit 10 \
    --output_folder "./results/debug_run"

```

## Output Results

After execution, the following key files will be generated in `output_folder`:

* `res.jsonl`: Contains localization results for each Issue (list of files, classes, and functions).
* `loc_trajs.jsonl`: Detailed localization trajectories, including every thought, tool call, and observation of the agents.
* `localize.log`: Detailed log of the running process for debugging.

## Troubleshooting

* **Index Loading Failed**: Please check if `GRAPH_INDEX_DIR` and `BM25_INDEX_DIR` in the `.env` file point to the correct **directories** (not the files themselves).
* **API Connection Error**: Please check if `OPENAI_API_BASE` and `OPENAI_API_KEY` are correct, and verify your network connection.
* **Git Error**: Ensure `REPO_CACHE_DIR` has write permissions and that `git` is installed on your system.

## License

MIT License