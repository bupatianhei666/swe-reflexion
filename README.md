# SWE-Reflexion

**基于大语言模型 API 的多智能体协作代码缺陷定位工具**

SWE-Reflexion 是一个利用大语言模型（如 GPT-4, DeepSeek 等）API 驱动的自动化代码缺陷定位框架。与传统的单向搜索不同，本项目采用 **多智能体协作（Multi-Agent Collaboration）** 机制，通过主智能体（Explorer）、批判智能体（Critic）和反思智能体（Reflector）的交互，结合 **图索引（Graph Index）** 和 **BM25 索引** 增强检索能力，实现高精度的缺陷定位。


## 核心特性

- **多智能体协同**: 
  - **Explorer**: 执行代码搜索、阅读和推理。
  - **Critic**: 审查 Explorer 的路径，指出无效探索。
  - **Reflector**: 在陷入僵局时总结经验，调整搜索策略。
- **混合检索增强**: 
  - 集成 **代码属性图（Code Property Graph）索引** 用于结构化导航。
  - 集成 **BM25 稀疏索引** 用于精确关键词匹配。
  - 集成 **向量索引** 用于语义搜索。
- **自我反思与回溯**: 支持基于历史交互的自我纠错，避免在错误路径上浪费 Token。

## 环境准备

### 1. 安装依赖

请确保 Python 版本 >= 3.9，并安装项目依赖：

```bash
pip install -r requirements.txt

```

### 2. 配置环境变量 (关键步骤)

在项目根目录下创建一个 `.env` 文件，并严格按照以下格式配置路径和密钥。这是系统运行的基础。

**`.env` 文件示例：**

```ini
# ==============================
# 模型 API 配置 (必须)
# ==============================
# 兼容 OpenAI 格式的 API 接口地址
OPENAI_API_BASE="[https://api.deepseek.com/v1](https://api.deepseek.com/v1)"
# 您的 API 密钥
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 指定使用的模型名称 (需与 API 供应商支持的名称一致)
MODEL_NAME="deepseek-chat"

# ==============================
# 索引数据路径配置 (必须)
# ==============================
# 代码属性图索引目录 (用于结构化遍历 explore_tree_structure)
GRAPH_INDEX_DIR="/path/to/your/project/index_data/graph_index"

# BM25 稀疏索引目录 (用于关键词搜索 keyword_search)
BM25_INDEX_DIR="/path/to/your/project/index_data/bm25_index"

# 向量索引目录 (用于语义搜索 semantic_search)
VECTOR_INDEX_DIR="/path/to/your/project/index_data/vector_index"

# 仓库缓存目录 (用于 git 操作和源码读取)
REPO_CACHE_DIR="./playground/repo"

```

> **提示**: 请确保上述索引目录中已包含针对目标数据集（如 SWE-bench）预构建好的索引文件。

## 快速开始

### 运行定位脚本

使用 `auto_search_main_single.py` 启动定位任务。该脚本会自动读取 `.env` 中的配置连接模型和索引。

```bash
python auto_search_main_single.py \
    --output_folder ./results_api_test \
    --num_samples 1 \
    --max_attempt_num 2 \
    --use_function_calling

```

### 常用参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | (读取.env) | 指定使用的 LLM 模型名称 (也可通过命令行覆盖) |
| `--output_folder` | `lite_raw_res` | 结果输出目录 |
| `--dataset` | `czlll/SWE-bench_Lite` | 评估使用的数据集 |
| `--eval_n_limit` | 274 | 限制评估的样本数量 |
| `--critic_agent` | `True` | 是否启用批判智能体 (建议开启) |
| `--reflection_agent` | `True` | 是否启用反思智能体 (建议开启) |
| `--use_function_calling` | `True` | 启用 Function Calling 格式 (推荐) |
| `--enable_semantic_search` | `True` | 是否允许使用语义搜索工具 |

### 运行示例

**1. 使用 DeepSeek-V3 进行完整测试：**

```bash
python auto_search_main_single.py \
    --model "deepseek-chat" \
    --output_folder "./results/deepseek_v3" \
    --use_function_calling \
    --critic_agent True

```

**2. 仅运行前 10 个样本进行快速验证：**

```bash
python auto_search_main_single.py \
    --eval_n_limit 10 \
    --output_folder "./results/debug_run"

```

## 输出结果

运行完成后，`output_folder` 中将生成以下关键文件：

* `res.jsonl`: 包含每个 Issue 的定位结果（文件、类、函数列表）。
* `loc_trajs.jsonl`: 详细的定位轨迹，包含智能体的每一次思考、工具调用和观察结果。
* `localize.log`: 运行过程的详细日志，用于调试。

## 故障排除

* **索引加载失败**: 请检查 `.env` 文件中的 `GRAPH_INDEX_DIR` 和 `BM25_INDEX_DIR` 是否指向了包含索引文件的正确**目录**（而不是文件本身）。
* **API 连接错误**: 请检查 `OPENAI_API_BASE` 和 `OPENAI_API_KEY` 是否正确，以及网络连接是否正常。
* **Git 错误**: 确保 `REPO_CACHE_DIR` 有写入权限，且系统已安装 `git`。

## 许可证

MIT License

```