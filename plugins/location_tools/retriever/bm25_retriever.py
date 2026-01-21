import os
import pickle
import Stemmer
import fnmatch
import mimetypes
from typing import Dict, List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from repo_index.index.epic_split import EpicSplitter

from dependency_graph import RepoEntitySearcher
from dependency_graph.traverse_graph import is_test_file
from dependency_graph.build_graph import (
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
)

import warnings

warnings.simplefilter('ignore', FutureWarning)

NTYPES = [
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_CLASS,
]


# 从指定的代码仓库中构建一个基于BM25的检索器
def build_code_retriever_from_repo(repo_path,
                                   similarity_top_k=10,
                                   min_chunk_size=100,
                                   chunk_size=500,
                                   max_chunk_size=2000,
                                   hard_token_limit=2000,
                                   max_chunks=200,
                                   persist_path=None,
                                   show_progress=False,
                                   ):
    # print(repo_path)
    # Only extract file name and type to not trigger unnecessary embedding jobs
    def file_metadata_func(file_path: str) -> Dict:
        # print(file_path)
        file_path = file_path.replace(repo_path, '')
        if file_path.startswith('/'):
            file_path = file_path[1:]
        # Windows 路径兼容
        if file_path.startswith('\\'):
            file_path = file_path[1:]
        # 统一路径分隔符
        file_path = file_path.replace('\\', '/')

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

    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                '**/test/**',
                '**/tests/**',
                '**/test_*.py',
                '**/*_test.py',
            ],
            file_metadata=file_metadata_func,
            filename_as_id=True,
            required_exts=['.py'],  # TODO: Shouldn't be hardcoded and filtered
            recursive=True,
        )
        docs = reader.load_data()
    except ValueError as e:
        # SimpleDirectoryReader 在没有找到文件时会抛出 ValueError
        print(f"警告: 仓库 {repo_path} 中没有找到符合条件的 .py 文件: {e}")
        docs = []

    # ========== 添加诊断信息 ==========
    # 统计仓库中的 .py 文件数量（用于诊断）
    all_py_files = []
    test_py_files = []
    for root, dirs, files in os.walk(repo_path):
        # 跳过 .git 目录
        if '.git' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                rel_path = rel_path.replace('\\', '/')
                all_py_files.append(rel_path)
                # 检查是否是测试文件
                test_patterns = ['test/', 'tests/', 'test_', '_test.py']
                if any(p in rel_path.lower() for p in test_patterns):
                    test_py_files.append(rel_path)

    non_test_count = len(all_py_files) - len(test_py_files)

    # 检查是否有文档
    if not docs:
        print(f"警告: 仓库 {repo_path} 中没有找到可索引的文档")
        print(f"  诊断信息: 总 .py 文件: {len(all_py_files)}, 测试文件: {len(test_py_files)}, 非测试文件: {non_test_count}")
        if non_test_count > 0:
            print(f"  ⚠️ 异常: 有 {non_test_count} 个非测试文件但未被读取，可能是路径或编码问题")
            # 显示前几个未被读取的文件
            non_test_files = [f for f in all_py_files if not any(p in f.lower() for p in ['test/', 'tests/', 'test_', '_test.py'])]
            for f in non_test_files[:5]:
                print(f"    - {f}")

        # 如果指定了持久化路径，创建一个空标记
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            empty_marker = os.path.join(persist_path, '.empty_index')
            with open(empty_marker, 'w', encoding='utf-8') as f:
                f.write(f"No indexable documents found in {repo_path}\n")
                f.write(f"Total .py files: {len(all_py_files)}\n")
                f.write(f"Test files: {len(test_py_files)}\n")
                f.write(f"Non-test files: {non_test_count}\n")
        return None

    print(f"  加载文档: {len(docs)} 个（总 .py: {len(all_py_files)}, 测试: {len(test_py_files)}）")

    splitter = EpicSplitter(
        min_chunk_size=min_chunk_size,
        chunk_size=chunk_size,
        max_chunk_size=max_chunk_size,
        hard_token_limit=hard_token_limit,
        max_chunks=max_chunks,
        repo_path=repo_path,
    )
    prepared_nodes = splitter.get_nodes_from_documents(docs, show_progress=show_progress)

    # ========== 关键修复：检查 prepared_nodes 是否为空 ==========
    if not prepared_nodes:
        print(f"警告: 仓库 {repo_path} 分割后没有节点")
        print(f"  诊断信息: 加载了 {len(docs)} 个文档，但分割后为空")
        print(f"  ⚠️ 可能原因: 文件内容过短或分割参数不合适")
        # 如果指定了持久化路径，创建一个空标记
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
            empty_marker = os.path.join(persist_path, '.empty_index')
            with open(empty_marker, 'w', encoding='utf-8') as f:
                f.write(f"No nodes after splitting in {repo_path}\n")
                f.write(f"Documents loaded: {len(docs)}\n")
        return None

    # We can pass in the index, docstore, or list of nodes to create the retriever
    retriever = BM25Retriever.from_defaults(
        nodes=prepared_nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    if persist_path:
        retriever.persist(persist_path)
    return retriever


def build_retriever_from_persist_dir(path: str):
    """从持久化目录加载 BM25 检索器"""
    # 检查是否是空索引
    empty_marker = os.path.join(path, '.empty_index')
    if os.path.exists(empty_marker):
        print(f"警告: {path} 是空索引，返回 None")
        return None

    # 自定义加载过程，确保使用UTF-8编码
    # import os
    import json
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.retrievers.bm25.base import DEFAULT_PERSIST_FILENAME
    import bm25s

    # 修改bm25s.BM25.load方法中的文件打开方式
    original_open = open

    def utf8_open(file, mode='r', *args, **kwargs):
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return original_open(file, mode, *args, **kwargs)

    # 临时替换open函数
    import builtins
    original_open = builtins.open
    builtins.open = utf8_open

    try:
        retriever = BM25Retriever.from_persist_dir(path)
    finally:
        # 恢复原始open函数
        builtins.open = original_open

    return retriever

# def build_retriever_from_persist_dir(path: str):
#     """从持久化目录加载 BM25 检索器"""
#     # 检查是否是空索引
#     empty_marker = os.path.join(path, '.empty_index')
#     if os.path.exists(empty_marker):
#         print(f"警告: {path} 是空索引，返回 None")
#         return None
#
#     import json
#     import inspect
#     from llama_index.retrievers.bm25 import BM25Retriever
#     from llama_index.retrievers.bm25.base import DEFAULT_PERSIST_FILENAME
#     from llama_index.core.schema import TextNode
#     import bm25s
#
#     # 临时替换open函数以使用UTF-8编码
#     original_open = open
#     def utf8_open(file, mode='r', *args, **kwargs):
#         if 'b' not in mode and 'encoding' not in kwargs:
#             kwargs['encoding'] = 'utf-8'
#         return original_open(file, mode, *args, **kwargs)
#
#     import builtins
#     builtins.open = utf8_open
#
#     try:
#         # 1. 加载BM25索引
#         bm25 = bm25s.BM25.load(path, mmap=True)
#
#         # 2. 加载retriever配置
#         retriever_path = os.path.join(path, DEFAULT_PERSIST_FILENAME)
#         with utf8_open(retriever_path, 'r') as f:
#             retriever_data = json.load(f)
#
#         # 3. 从 corpus.jsonl 加载 nodes（关键修复）
#         corpus_path = os.path.join(path, 'corpus.jsonl')
#         nodes = []
#
#         if os.path.exists(corpus_path):
#             with utf8_open(corpus_path, 'r') as f:
#                 for line in f:
#                     if line.strip():
#                         node_dict = json.loads(line)
#                         # 重建 TextNode 对象
#                         node = TextNode(
#                             text=node_dict.get('text', ''),
#                             id_=node_dict.get('id_', ''),
#                             metadata=node_dict.get('metadata', {}),
#                         )
#                         # 恢复其他可能的属性
#                         if 'embedding' in node_dict:
#                             node.embedding = node_dict['embedding']
#                         nodes.append(node)
#             print(f"成功加载 {len(nodes)} 个节点")
#         else:
#             print(f"警告: 未找到 corpus.jsonl 文件: {corpus_path}")
#             return None
#
#         # 4. 获取BM25Retriever.__init__接受的参数
#         init_signature = inspect.signature(BM25Retriever.__init__)
#         valid_params = set(init_signature.parameters.keys()) - {'self', 'existing_bm25', 'nodes'}
#
#         # 5. 过滤掉不兼容的参数（排除 nodes，因为单独处理）
#         filtered_data = {k: v for k, v in retriever_data.items()
#                          if k in valid_params and k != 'nodes'}
#
#         # 记录被过滤的参数
#         filtered_out = set(retriever_data.keys()) - set(filtered_data.keys()) - {'nodes'}
#         if filtered_out:
#             print(f"警告: 过滤了不兼容的参数: {filtered_out}")
#
#         # 6. 创建retriever（显式传递 nodes）
#         retriever = BM25Retriever(
#             nodes=nodes,  # 显式传递加载的节点
#             existing_bm25=bm25,
#             **filtered_data
#         )
#
#         print(f"✅ BM25 检索器加载成功，包含 {len(nodes)} 个文档")
#
#     except Exception as e:
#         print(f"❌ 加载 BM25 检索器失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
#     finally:
#         # 恢复原始open函数
#         builtins.open = original_open
#
#     return retriever

def build_module_retriever_from_graph(graph_path: Optional[str] = None,
                                      entity_searcher: Optional[RepoEntitySearcher] = None,
                                      search_scope: str = 'all',
                                      # enum = {'function', 'class', 'file', 'all'}
                                      similarity_top_k: int = 10,

                                      ):
    assert search_scope in NTYPES or search_scope == 'all'
    assert graph_path or isinstance(entity_searcher, RepoEntitySearcher)

    if graph_path:
        G = pickle.load(open(graph_path, "rb"))
        entity_searcher = RepoEntitySearcher(G)
    else:
        G = entity_searcher.G

    selected_nodes = list()
    for nid in G:
        if is_test_file(nid): continue

        ndata = entity_searcher.get_node_data([nid])[0]
        ndata['nid'] = nid  # add `nid` property
        if search_scope == 'all':  # and ndata['type'] in NTYPES[2:]
            selected_nodes.append(ndata)
        elif ndata['type'] == search_scope:
            selected_nodes.append(ndata)

    # 检查是否有节点
    if not selected_nodes:
        print(f"警告: 图中没有找到符合条件的节点")
        return None

    # initialize node parser
    splitter = SimpleFileNodeParser()
    documents = [Document(text=t['nid']) for t in selected_nodes]
    nodes = splitter.get_nodes_from_documents(documents)

    # We can pass in the index, docstore, or list of nodes to create the retriever
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    return retriever