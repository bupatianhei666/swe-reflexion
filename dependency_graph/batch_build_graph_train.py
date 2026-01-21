"""
batch_build_graph.py - 批量构建代码语义图

修改点：
1. 从本地 JSONL 文件加载数据（而非 Huggingface）
2. 支持仓库共享机制（同一 repo 不同 instance_id 共享仓库）
3. **关键修改**：按 repo 分组，每个进程负责一个完整的 repo
   - 同一 repo 内的 instance 串行处理（避免 Git 锁冲突）
   - 不同 repo 之间并行处理（提高效率）

使用方法:
python batch_build_graph.py \
    --input_file merged_localization_dataset.jsonl \
    --repo_cache_dir ./playground \
    --index_dir ./index_data \
    --num_processes 8
"""

import argparse
import json
import os
import pickle
import time
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch.multiprocessing as mp
import os.path as osp
from datasets import load_dataset

from dependency_graph.build_graph import build_graph, VERSION
from util.benchmark.setup_repo import setup_repo
import dotenv
dotenv.load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# Git 仓库管理器（单进程版本，无需锁）
# ═══════════════════════════════════════════════════════════════════════════════

class GitManager:
    """Git 仓库管理器 - 单进程内使用，无并发冲突"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _safe_repo_name(self, repo_name: str) -> str:
        """将 repo 名称转换为安全的目录名: django/django -> django__django"""
        return repo_name.replace("/", "__")

    def get_repo_dir(self, repo_name: str) -> str:
        """获取仓库目录路径"""
        safe_name = self._safe_repo_name(repo_name)
        return os.path.join(self.cache_dir, safe_name)

    def run_git_command(self, cmd: List[str], cwd: str = None, timeout: int = 300) -> Tuple[bool, str]:
        """执行 Git 命令"""
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'
        try:
            result = subprocess.run(
                cmd, cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=True,
                timeout=timeout,
                text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def ensure_repo_cloned(self, repo_name: str) -> bool:
        """确保仓库已克隆"""
        repo_dir = self.get_repo_dir(repo_name)

        # 检查仓库是否已存在
        if os.path.exists(os.path.join(repo_dir, '.git')):
            print(f"    仓库已存在: {repo_name}")
            return True

        os.makedirs(repo_dir, exist_ok=True)
        print(f"    开始克隆仓库: {repo_name}")

        clone_url = f"https://github.com/{repo_name}"
        cmd = ["git", "clone", clone_url, repo_dir]
        success, msg = self.run_git_command(cmd, timeout=600)

        if success:
            print(f"    仓库克隆成功: {repo_name}")
            return True
        else:
            print(f"    仓库克隆失败: {repo_name}, 错误: {msg}")
            # 清理失败的目录
            try:
                if os.path.exists(repo_dir) and not os.path.exists(os.path.join(repo_dir, '.git')):
                    shutil.rmtree(repo_dir)
            except:
                pass
            return False

    def checkout_commit(self, repo_name: str, commit: str) -> bool:
        """切换到指定 commit"""
        repo_dir = self.get_repo_dir(repo_name)

        if not os.path.exists(os.path.join(repo_dir, '.git')):
            print(f"    仓库不存在: {repo_dir}")
            return False

        # 清理可能残留的锁文件
        lock_file = os.path.join(repo_dir, '.git', 'index.lock')
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print(f"    清理残留锁文件: {lock_file}")
            except:
                pass

        # 先 fetch 确保有最新的 commits
        self.run_git_command(["git", "fetch", "--all"], cwd=repo_dir, timeout=300)

        # 清理工作区
        self.run_git_command(["git", "clean", "-fd"], cwd=repo_dir)
        self.run_git_command(["git", "reset", "--hard"], cwd=repo_dir)

        # checkout 到指定 commit
        success, msg = self.run_git_command(
            ["git", "checkout", "-f", commit],
            cwd=repo_dir,
            timeout=120
        )

        if success:
            print(f"    成功切换到 commit: {commit[:8]}")
            return True
        else:
            print(f"    切换 commit 失败: {commit[:8]}, 错误: {msg}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载工具
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_from_jsonl(file_path: str) -> List[Dict]:
    """从 JSONL 文件加载数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_instances_by_repo(instances: List[Dict]) -> Dict[str, List[Dict]]:
    """按 repo 分组 instances"""
    grouped = defaultdict(list)
    for instance in instances:
        repo = instance.get('repo', '')
        if repo:
            grouped[repo].append(instance)
    return dict(grouped)


def list_folders(path):
    """列出目录下的所有文件夹"""
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


# ═══════════════════════════════════════════════════════════════════════════════
# 工作进程函数（关键修改：每个进程处理完整的 repo）
# ═══════════════════════════════════════════════════════════════════════════════

def run_repo_worker(
        rank: int,
        repo_queue,  # 队列中的元素是 (repo_name, [instance1, instance2, ...])
        repo_cache_dir: str,
        index_dir: str,
):
    """
    工作进程函数 - 每个进程负责处理完整的 repo

    关键设计：
    - 从队列获取一个 repo 及其所有 instance
    - 串行处理该 repo 下的所有 instance（避免并发冲突）
    - 处理完一个 repo 后再获取下一个
    """
    git_manager = GitManager(repo_cache_dir)

    while True:
        try:
            repo_name, instances = repo_queue.get_nowait()
        except Exception:
            # 队列为空，退出循环
            break

        print(f'\n[进程 {rank}] ========== 开始处理仓库: {repo_name} ({len(instances)} 个实例) ==========')

        # 1. 确保仓库已克隆（只需要做一次）
        if not git_manager.ensure_repo_cloned(repo_name):
            print(f'[进程 {rank}] 克隆仓库失败: {repo_name}，跳过该仓库所有实例')
            continue

        repo_dir = git_manager.get_repo_dir(repo_name)

        # 2. 串行处理该 repo 下的所有 instance
        for idx, instance in enumerate(instances):
            instance_id = instance['instance_id']
            base_commit = instance.get('base_commit', '')

            output_file = os.path.join(index_dir, f'{instance_id}.pkl')

            # 检查是否已处理
            if os.path.exists(output_file):
                print(f'[进程 {rank}] [{idx+1}/{len(instances)}] {instance_id} 已处理，跳过')
                continue

            print(f'[进程 {rank}] [{idx+1}/{len(instances)}] 开始处理: {instance_id} (commit: {base_commit[:8] if base_commit else "N/A"})')

            try:
                # 2.1 切换到指定 commit
                if base_commit:
                    if not git_manager.checkout_commit(repo_name, base_commit):
                        print(f'[进程 {rank}] 切换 commit 失败，跳过: {instance_id}')
                        continue

                # 2.2 构建代码图
                print(f'[进程 {rank}] 开始构建代码图: {instance_id}')
                G = build_graph(repo_dir, global_import=True)

                # 2.3 保存代码图
                with open(output_file, 'wb') as f:
                    pickle.dump(G, f)
                print(f'[进程 {rank}] 成功处理: {instance_id}')

            except Exception as e:
                print(f'[进程 {rank}] 处理 {instance_id} 时出错: {e}')
                import traceback
                traceback.print_exc()

        print(f'[进程 {rank}] ========== 完成仓库: {repo_name} ==========\n')


def run_original(
        rank: int,
        repo_queue,
        repo_path: str,
        out_path: str,
        download_repo: bool = False,
        instance_data: Dict = None
):
    """
    原有的工作进程函数（兼容原有方式）
    """
    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            break

        output_file = f'{osp.join(out_path, repo_name)}.pkl'
        if osp.exists(output_file):
            print(f'[{rank}] {repo_name} already processed, skipping.')
            continue

        if download_repo:
            repo_base_dir = str(osp.join(repo_path, str(rank)))
            os.makedirs(repo_base_dir, exist_ok=True)
            try:
                repo_dir = setup_repo(
                    instance_data=instance_data[repo_name],
                    repo_base_dir=repo_base_dir,
                    dataset=None
                )
            except subprocess.CalledProcessError as e:
                print(f'[{rank}] Error checkout commit {repo_name}: {e}')
                continue
        else:
            repo_dir = osp.join(repo_path, repo_name)

        print(f'Start process {repo_name}')
        try:
            G = build_graph(repo_dir, global_import=True)
            with open(output_file, 'wb') as f:
                pickle.dump(G, f)
            print(f'[{rank}] Processed {repo_name}')
        except Exception as e:
            print(f'[{rank}] Error processing {repo_name}: {e}')


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="批量构建代码语义图")

    # ========== 新增：JSONL 输入模式 ==========
    parser.add_argument("--input_file", type=str, default=r"D:\projects_eng\loc_proj\loc_agent\LocAgent\merged_localization_dataset.jsonl",
                        help="训练数据集 JSONL 文件路径（新模式）")
    parser.add_argument("--repo_cache_dir", type=str, default=r"D:\projects_eng\loc_proj\loc_agent\LocAgent\playground\repo",
                        help="仓库缓存目录（新模式，支持仓库共享）")

    # ========== 原有参数（兼容原有模式）==========
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite",
                        help="Huggingface 数据集名称（原有模式）")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--download_repo', action='store_true', default=False,
                        help='是否从 GitHub 下载仓库（原有模式）')
    parser.add_argument('--repo_path', type=str, default='playground/build_graph',
                        help='仓库下载目录（原有模式）')
    parser.add_argument('--instance_id_path', type=str, default='',
                        help='指定 instance_id 列表文件')

    # ========== 通用参数 ==========
    parser.add_argument('--num_processes', type=int, default=5,
                        help='并行进程数')
    parser.add_argument('--index_dir', type=str, default=r"D:\projects_eng\loc_proj\loc_agent\LocAgent\index_data\loc-bench",
                        help='代码图索引输出目录')

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════════
    # 判断运行模式：新模式（JSONL）或 原有模式（Huggingface）
    # ═══════════════════════════════════════════════════════════════════════════

    if args.input_file and os.path.exists(args.input_file):
        # ========== 新模式：从 JSONL 文件加载 ==========
        print("=" * 60)
        print("运行模式: 从 JSONL 文件加载（支持仓库共享）")
        print("=" * 60)

        # 设置索引输出目录
        args.index_dir = os.path.join(args.index_dir, f'graph_index_{VERSION}')
        os.makedirs(args.index_dir, exist_ok=True)

        # 加载数据
        print(f"加载数据: {args.input_file}")
        all_instances = load_dataset_from_jsonl(args.input_file)
        print(f"共加载 {len(all_instances)} 个实例")

        # 按 repo 分组
        grouped = group_instances_by_repo(all_instances)
        print(f"共 {len(grouped)} 个唯一仓库")

        # 统计每个 repo 的 instance 数量
        repo_stats = [(repo, len(instances)) for repo, instances in grouped.items()]
        repo_stats.sort(key=lambda x: x[1], reverse=True)
        print("\n仓库统计（按实例数排序）:")
        for repo, count in repo_stats[:10]:
            print(f"  {repo}: {count} 个实例")
        if len(repo_stats) > 10:
            print(f"  ... 还有 {len(repo_stats) - 10} 个仓库")

        # 过滤已处理的实例，重新分组
        filtered_grouped = {}
        total_pending = 0
        for repo_name, instances in grouped.items():
            pending_instances = []
            for instance in instances:
                instance_id = instance['instance_id']
                output_file = os.path.join(args.index_dir, f'{instance_id}.pkl')
                if not os.path.exists(output_file):
                    pending_instances.append(instance)

            if pending_instances:
                filtered_grouped[repo_name] = pending_instances
                total_pending += len(pending_instances)

        print(f"\n待处理: {total_pending} 个实例，分布在 {len(filtered_grouped)} 个仓库")

        if total_pending == 0:
            print("所有实例已处理完成")
            return

        # 创建任务队列 - 每个任务是一个完整的 repo
        manager = mp.Manager()
        repo_queue = manager.Queue()

        # 将 repo 及其 instances 放入队列
        # 按 instance 数量排序，大的 repo 先处理（负载均衡）
        sorted_repos = sorted(filtered_grouped.items(), key=lambda x: len(x[1]), reverse=True)
        for repo_name, instances in sorted_repos:
            repo_queue.put((repo_name, instances))

        print(f"任务队列: {repo_queue.qsize()} 个仓库")

        # 调整进程数，不超过 repo 数量
        actual_processes = min(args.num_processes, len(filtered_grouped))
        print(f"启动进程数: {actual_processes}")

        # 启动多进程
        start_time = time.time()

        mp.spawn(
            run_repo_worker,
            nprocs=actual_processes,
            args=(repo_queue, args.repo_cache_dir, args.index_dir),
            join=True
        )

        end_time = time.time()
        print(f'\n总耗时: {end_time - start_time:.3f}s')

    else:
        # ========== 原有模式：从 Huggingface 加载 ==========
        print("=" * 60)
        print("运行模式: 从 Huggingface 加载（原有方式）")
        print("=" * 60)

        dataset_name = args.dataset.split('/')[-1]
        args.index_dir = f'{args.index_dir}/{dataset_name}/graph_index_{VERSION}/'
        os.makedirs(args.index_dir, exist_ok=True)

        # 加载数据
        if args.download_repo:
            selected_instance_data = {}
            bench_data = load_dataset(args.dataset, split=args.split)

            if args.instance_id_path and osp.exists(args.instance_id_path):
                with open(args.instance_id_path, 'r') as f:
                    repo_folders = json.loads(f.read())
                for instance in bench_data:
                    if instance['instance_id'] in repo_folders:
                        selected_instance_data[instance['instance_id']] = instance
            else:
                repo_folders = []
                for instance in bench_data:
                    repo_folders.append(instance['instance_id'])
                    selected_instance_data[instance['instance_id']] = instance
        else:
            if args.instance_id_path and osp.exists(args.instance_id_path):
                with open(args.instance_id_path, 'r') as f:
                    repo_folders = json.loads(f.read())
            else:
                repo_folders = list_folders(args.repo_path)
            selected_instance_data = None

        os.makedirs(args.repo_path, exist_ok=True)

        # 创建任务队列
        manager = mp.Manager()
        queue = manager.Queue()
        for repo in repo_folders:
            queue.put(repo)

        start_time = time.time()

        # 启动多进程
        mp.spawn(
            run_original,
            nprocs=args.num_processes,
            args=(queue, args.repo_path, args.index_dir,
                  args.download_repo, selected_instance_data),
            join=True
        )

        end_time = time.time()
        print(f'Total Execution time = {end_time - start_time:.3f}s')


if __name__ == '__main__':
    main()