import argparse
import json
import mimetypes
import os
import pickle
import time
from pathlib import Path
import torch.multiprocessing as mp
import os.path as osp
from datasets import load_dataset
from util.benchmark.setup_repo import setup_repo
from repo_index.repository import FileRepository
from repo_index.index.code_index import CodeIndex
from repo_index.index.settings import IndexSettings
from repo_index.index.simple_faiss import SimpleFaissVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
import logging
from logging.handlers import RotatingFileHandler
import fnmatch
import faiss

# 设置日志
# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "vector_build.log")

# 配置日志处理器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# 创建文件处理器 (使用RotatingFileHandler以防日志文件过大)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# 添加处理器到logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 向量索引保存目录
VECTOR_INDEX_DIR = os.environ.get("VECTOR_INDEX_DIR", "index_data/vector_index")
# 图索引目录，用于获取已处理的项目列表
GRAPH_INDEX_DIR = os.environ.get("GRAPH_INDEX_DIR", "index_data/graph_index")

def list_folders(path):
    """列出指定路径下的所有文件夹"""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_embed_model():
    """获取嵌入模型"""
    return DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        embed_batch_size=10  # 设置每批处理的文本数量为16
    )

def file_metadata_func(file_path, repo_path):
    """生成文件元数据"""
    file_path = file_path.replace(repo_path, '')
    if file_path.startswith('/'):
        file_path = file_path[1:]
    elif file_path.startswith('\\'):
        file_path = file_path[1:]

    test_patterns = [
        '**/test/**',
        '**/tests/**',
        '**/test_*.py',
        '**/*_test.py',
    ]
    category = (
        'test'
        if any(fnmatch.fnmatch(file_path, pattern) for pattern in test_patterns)
        else 'implementation'
    )

    return {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_type': mimetypes.guess_type(file_path)[0],
        'category': category,
    }

def build_vector_index(repo_dir, instance_id, persist_dir):
    """为单个代码仓库构建向量索引并持久化"""
    try:
        # 创建文件仓库
        file_repo = FileRepository(repo_dir)

        # 创建索引设置
        settings = IndexSettings(
            embed_model="text-embedding-v3",
            dimensions=1024
        )

        # 获取embedding模型
        embed_model = get_embed_model()

        # 创建向量存储和文档存储
        # 创建 Faiss 索引
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(settings.dimensions))
        vector_store = SimpleFaissVectorStore(faiss_index)
        docstore = SimpleDocumentStore()

        # 创建代码索引
        code_index = CodeIndex(
            file_repo=file_repo,
            vector_store=vector_store,
            docstore=docstore,
            embed_model=embed_model,
            settings=settings,
        )

        # 查找所有Python文件
        python_files = file_repo.find_files(["**/*.py"])
        logger.info(f"Found {len(python_files)} Python files in {instance_id}")

        # 使用SimpleDirectoryReader读取文件
        all_files = []
        for file_path in python_files:
            full_path = os.path.join(repo_dir, file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                all_files.append(full_path)

        if not all_files:
            logger.warning(f"No Python files found in {repo_dir}")
            return False

        # 直接使用CodeIndex的run_ingestion方法处理文档
        # 这个方法会处理文件、创建节点、计算嵌入并添加到向量存储中
        code_index.run_ingestion(
            repo_path=repo_dir,
            input_files=all_files,
        )
        logger.info(f"Processed files from {instance_id}")

        # 持久化索引
        code_index.persist(persist_dir)
        logger.info(f"Persisted vector index for {instance_id} to {persist_dir}")

        return True
    except Exception as e:
        logger.error(f"Error building vector index for {instance_id}: {e}")
        return False

def run(rank, repo_queue, repo_path, out_path, download_repo=False, instance_data=None):
    """处理队列中的代码仓库"""
    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            # 队列为空
            break

        output_dir = osp.join(out_path, repo_name)
        if osp.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.info(f'[{rank}] {repo_name} already processed, skipping.')
            continue

        if download_repo:
            # 获取进程特定的基础目录
            repo_base_dir = osp.join(repo_path, str(rank))
            os.makedirs(repo_base_dir, exist_ok=True)
            # 克隆并检查实际仓库
            try:
                repo_dir = setup_repo(instance_data=instance_data[repo_name],
                                      repo_base_dir=repo_base_dir,
                                      dataset=None)

                # 构建向量索引
                success = build_vector_index(repo_dir, repo_name, output_dir)
                if success:
                    logger.info(f'[{rank}] Processed {repo_name}')
                else:
                    logger.error(f'[{rank}] Failed to process {repo_name}')
            except Exception as e:
                logger.error(f'[{rank}] Error processing {repo_name}: {e}')
        else:
            # 使用已有的仓库
            repo_dir = osp.join(repo_path, repo_name)
            if not osp.exists(repo_dir):
                logger.warning(f'[{rank}] {repo_dir} does not exist, skipping.')
                continue

            try:
                # 添加：切换到正确的代码分支
                from util.benchmark.git_repo_manager import checkout_commit
                base_commit = instance_data[repo_name]['base_commit']
                logger.info(f'[{rank}] Checking out commit {base_commit} for {repo_name}')
                checkout_commit(repo_dir, base_commit)

                # 构建向量索引
                success = build_vector_index(repo_dir, repo_name, output_dir)
                if success:
                    logger.info(f'[{rank}] Processed {repo_name}')
                else:
                    logger.error(f'[{rank}] Failed to process {repo_name}')
            except Exception as e:
                logger.error(f'[{rank}] Error processing {repo_name}: {e}')



def main():
    parser = argparse.ArgumentParser(description='批量构建向量索引')
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite",
                        help="数据集名称")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集分割")
    parser.add_argument('--num_processes', type=int, default=1,
                        help='并行处理的进程数')
    parser.add_argument('--download_repo', action='store_true', default=False,
                        help='是否在索引前下载代码库到`repo_path`')
    parser.add_argument('--repo_path', type=str, default='playground/repo',
                        help='您计划拉取或已经拉取代码库的目录')
    parser.add_argument('--index_dir', type=str, default='index_data',
                        help='生成的向量索引将保存的基础目录')
    parser.add_argument('--instance_id_path', type=str, default='',
                        help='包含要处理的实例ID列表的文件路径')
    args = parser.parse_args()

    # 创建输出目录
    dataset_name = args.dataset.split('/')[-1]
    out_path = osp.join(args.index_dir, dataset_name, 'vector_index')
    os.makedirs(out_path, exist_ok=True)

    # 加载数据集
    dataset_obj = load_dataset(args.dataset, split=args.split)

    # 先只生成前30个项目的向量索引
    dataset_obj = dataset_obj.select(range(0, 40))
    logger.info(f'加载数据集 {args.dataset}，包含 {len(dataset_obj)} 个实例')

    # 获取实例ID列表
    if args.instance_id_path and osp.exists(args.instance_id_path):
        with open(args.instance_id_path, 'r') as f:
            instance_ids = [line.strip() for line in f.readlines()]
    else:
        instance_ids = [item['instance_id'] for item in dataset_obj]

    # 创建实例数据字典
    instance_data = {item['instance_id']: item for item in dataset_obj}

    # 过滤实例ID
    filtered_instance_ids = []
    for instance_id in instance_ids:
        if instance_id in instance_data:
            filtered_instance_ids.append(instance_id)
        else:
            logger.warning(f'实例ID {instance_id} 不在数据集中，跳过')

    # 创建多进程队列
    repo_queue = mp.Queue()
    for instance_id in filtered_instance_ids:
        repo_queue.put(instance_id)

    # 启动多进程
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(
            target=run,
            args=(rank, repo_queue, args.repo_path, out_path, args.download_repo, instance_data)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    logger.info(f'所有实例的向量索引构建完成，保存在 {out_path}')

if __name__ == '__main__':
    main()