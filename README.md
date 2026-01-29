# SWE-Reflexion

SWE-Reflexion is an automated code bug localization framework driven by Large Language Model (LLM) APIs (e.g., GPT-4, DeepSeek). Unlike traditional one-way search methods, this project adopts a **Multi-Agent Collaboration** mechanism. Through the interaction of the main agent (Explorer), the critique agent (Critic), and the reflection agent (Reflector), combined with **Graph Index** and **BM25 Index** to enhance retrieval capabilities, it achieves high-precision bug localization.

## 📂 Project Structure

To help you navigate the repository, here is a description of the key directories and files:

* **`ablation_results/`**: Contains raw localization outputs for **Ablation Studies**. It includes different configurations such as `no_critic` and `no_reflect` to verify the effectiveness of the multi-agent components.
* **`result/`**: Stores the final experimental results of the full **SWE-Reflexion** pipeline.
* **`data/`**: Contains the **SWE-bench-Lite** dataset files used for evaluation.
* **`dependency_graph/`**: Scripts and logic for building and traversing the Code Property Graph (CPG).
* **`evaluation/`**: Evaluation scripts (e.g., `eval_metric.py`) to calculate metrics like Accuracy and MRR.
* **`repo_index/`**: Pre-built indexes (BM25, Graph, etc.) for the target repositories.
* **`scripts/`**: Utility scripts for data processing and batch experiment execution.
* **`auto_search_main_single.py`**: The entry point for running the localization framework on a single or batch of issues.
* **`res.ipynb`**: A Jupyter Notebook for result visualization and comparison of different experiment groups.

## 🚀 Core Features

* **Multi-Agent Collaboration**:
    * **Explorer**: Executes code search, reading, and reasoning.
    * **Critic**: Reviews the Explorer's path and points out invalid explorations.
    * **Reflector**: Summarizes experience and adjusts search strategies when stuck.
* **Hybrid Retrieval Enhancement**: Integrates **Graph Index** for structured navigation and **BM25 Sparse Index** for keyword matching.
* **Self-Reflection & Backtracking**: Supports self-correction to avoid wasting tokens on wrong paths.

## 🛠️ Prerequisites

### 1. Install Dependencies
Ensure Python version >= 3.9, then install the components:
```bash
pip install -r requirements.txt
# Note: For Jupyter support, please install ipywidgets
pip install ipywidgets tqdm

```

### 2. Configure Environment Variables (Critical)

Create a `.env` file in the root directory:

```ini
OPENAI_API_BASE="[https://api.deepseek.com/v1](https://api.deepseek.com/v1)"
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
MODEL_NAME="deepseek-chat"

GRAPH_INDEX_DIR="./index_data/graph_index"
BM25_INDEX_DIR="./index_data/bm25_index"
REPO_CACHE_DIR="./playground/repo"

```

## 📊 Quick Start

### Run Localization

```bash
python auto_search_main_single.py --output_folder ./results_test --num_samples 1 --use_function_calling

```

### Evaluate Results

You can use the provided Jupyter Notebook `res.ipynb` to analyze the outputs in `ablation_results/` and `result/`. It generates comparison tables for different agent configurations.

## 📝 License

MIT License
