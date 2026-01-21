import fnmatch
import json
import pickle
import pickle
import os
import re
from collections import defaultdict
from typing import List, Optional, Dict
import collections
from copy import deepcopy
import uuid
import networkx as nx
import torch
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datasets import load_dataset

from dependency_graph import RepoEntitySearcher, RepoDependencySearcher
from dependency_graph.build_graph import (
    build_graph,
    NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS, # EDGE_TYPE_INHERITS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS, 
    VALID_NODE_TYPES, VALID_EDGE_TYPES
)
from dependency_graph.traverse_graph import (
    is_test_file, traverse_tree_structure,
    traverse_graph_structure, traverse_json_structure,
)
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo as build_code_retriever,
    build_module_retriever_from_graph as build_module_retriever,
    build_retriever_from_persist_dir as load_retriever,
)
from plugins.location_tools.retriever.fuzzy_retriever import (
    fuzzy_retrieve_from_graph_nodes as fuzzy_retrieve
)
from plugins.location_tools.utils.result_format import QueryInfo, QueryResult
from plugins.location_tools.utils.util import (
    get_meta_data,
    find_matching_files_from_list,
    merge_intervals,
    GRAPH_INDEX_DIR,
    BM25_INDEX_DIR,
    DENSE_INDEX_DIR
)
from repo_index import FileRepository
from repo_index.index import SimpleFaissVectorStore, CodeIndex, IndexSettings
from util.benchmark.setup_repo import setup_repo
import subprocess
import logging
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
# from llama_index.embeddings.openai import OpenAIEmbedding

# from llama_index.embeddings.dashscope import (
#     DashScopeEmbedding,
#     DashScopeBatchTextEmbeddingModels,
#     DashScopeTextEmbeddingType, DashScopeTextEmbeddingModels,
# )

# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
# Settings.embed_model = DashScopeEmbedding(
#     model_name=DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V2,
#     text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
# )

logger = logging.getLogger(__name__)

CURRENT_ISSUE_ID: str | None = None
CURRENT_INSTANCE: dict | None = None
ALL_FILE: list | None = None
ALL_CLASS: list | None = None
ALL_FUNC: list | None = None

DP_GRAPH_ENTITY_SEARCHER: RepoEntitySearcher | None = None
DP_GRAPH_DEPENDENCY_SEARCHER: RepoDependencySearcher | None = None
DP_GRAPH: nx.MultiDiGraph | None = None

REPO_SAVE_DIR: str | None = None


def _normalize_node_id_for_graph(node_id: str, searcher) -> str:
    """
    尝试将节点ID标准化为图中实际存在的格式。
    处理Windows和Linux路径分隔符的差异。

    Args:
        node_id: 要查找的节点ID
        searcher: 图实体搜索器

    Returns:
        标准化后的节点ID（如果找到匹配），否则返回原始ID
    """
    if not node_id or not searcher:
        return node_id

    # 如果直接存在，返回原始ID
    if searcher.has_node(node_id):
        return node_id

    # 尝试将 \ 替换为 /
    normalized = node_id.replace('\\', '/')
    if normalized != node_id and searcher.has_node(normalized):
        return normalized

    # 尝试将 / 替换为 \
    normalized = node_id.replace('/', '\\')
    if normalized != node_id and searcher.has_node(normalized):
        return normalized

    return node_id

# 设置当前代码库的分析环境，加载或构建代码图，设置全局变量
def set_current_issue(instance_id: str = None, 
                      instance_data: dict = None,
                      dataset: str = "princeton-nlp/SWE-bench_Lite", split: str = "test", rank=0, logger=None):
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    assert instance_id or instance_data

    if instance_id:
        CURRENT_ISSUE_ID = instance_id
        CURRENT_INSTANCE = get_meta_data(CURRENT_ISSUE_ID, dataset, split)
    else:
        CURRENT_ISSUE_ID = instance_data['instance_id']
        CURRENT_INSTANCE = instance_data

    global REPO_SAVE_DIR
    # Generate a temperary folder and add uuid to avoid collision
    REPO_SAVE_DIR = os.path.join('playground', str(uuid.uuid4()))
    # assert playground doesn't exist
    assert not os.path.exists(REPO_SAVE_DIR), f"{REPO_SAVE_DIR} already exists"
    # create playground
    # os.makedirs(REPO_SAVE_DIR)
    
    # setup graph traverser
    global DP_GRAPH_ENTITY_SEARCHER, DP_GRAPH_DEPENDENCY_SEARCHER, DP_GRAPH
    graph_index_file = f"{GRAPH_INDEX_DIR}/{CURRENT_ISSUE_ID}.pkl"
    print(graph_index_file)
    if not os.path.exists(graph_index_file):
        # pull repo
        repo_dir = setup_repo(instance_data=CURRENT_INSTANCE, repo_base_dir=REPO_SAVE_DIR, dataset=None)
        # parse the repository:
        try:
            os.makedirs(GRAPH_INDEX_DIR, exist_ok=True)
            G = build_graph(repo_dir, global_import=True)
            with open(graph_index_file, 'wb') as f:
                pickle.dump(G, f)
            logging.info(f'[{rank}] Processed {CURRENT_ISSUE_ID}')
        except Exception as e:
            logging.error(f'[{rank}] Error processing {CURRENT_ISSUE_ID}: {e}')
    else:
        G = pickle.load(open(graph_index_file, "rb"))
        
    DP_GRAPH_ENTITY_SEARCHER = RepoEntitySearcher(G)
    DP_GRAPH_DEPENDENCY_SEARCHER = RepoDependencySearcher(G)
    DP_GRAPH = G
    if G is None:
        print("G is None")
    
    ALL_FILE = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_FILE)
    ALL_CLASS = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_CLASS)
    ALL_FUNC = DP_GRAPH_ENTITY_SEARCHER.get_all_nodes_by_type(NODE_TYPE_FUNCTION)

    if logger:
        logger.info(f'Rank = {rank}, set CURRENT_ISSUE_ID = {CURRENT_ISSUE_ID}')
        # 添加日志输出检查 ALL_FILE
        if ALL_FILE is None:
            logger.info("ALL_FILE is None.")
        elif not ALL_FILE: # 检查是否为空列表、空集合等
            logger.info("ALL_FILE is empty.")
        else:
            logger.info(f"ALL_FILE contains {len(ALL_FILE)} items.")

        # 添加日志输出检查 ALL_CLASS
        if ALL_CLASS is None:
            logger.info("ALL_CLASS is None.")
        elif not ALL_CLASS:
            logger.info("ALL_CLASS is empty.")
        else:
            logger.info(f"ALL_CLASS contains {len(ALL_CLASS)} items.")

        # 添加日志输出检查 ALL_FUNC
        if ALL_FUNC is None:
            logger.info("ALL_FUNC is None.")
        elif not ALL_FUNC:
            logger.info("ALL_FUNC is empty.")
        else:
            logger.info(f"ALL_FUNC contains {len(ALL_FUNC)} items.")



def reset_current_issue():
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE
    print("当前id")
    print(CURRENT_ISSUE_ID)
    CURRENT_ISSUE_ID = None
    CURRENT_INSTANCE = None

    global ALL_FILE, ALL_CLASS, ALL_FUNC
    ALL_FILE, ALL_CLASS, ALL_FUNC = None, None, None

    global REPO_SAVE_DIR
    # print(f"repo_save_dir: {REPO_SAVE_DIR}")
    if REPO_SAVE_DIR and os.path.exists(REPO_SAVE_DIR):
        # subprocess.run(
        #     ["rm", "-rf", REPO_SAVE_DIR], check=True
        # )
        # 删除该目录以及其子文件
        import shutil
        shutil.rmtree(REPO_SAVE_DIR)
    else:
        print(f"没找到 {REPO_SAVE_DIR}")
    REPO_SAVE_DIR = None


def get_current_issue_id():
    global CURRENT_ISSUE_ID
    return CURRENT_ISSUE_ID


def get_current_repo_modules():
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    return ALL_FILE, ALL_CLASS, ALL_FUNC


def get_current_issue_data():
    global CURRENT_INSTANCE
    return CURRENT_INSTANCE


def get_graph_entity_searcher() -> RepoEntitySearcher:
    global DP_GRAPH_ENTITY_SEARCHER
    return DP_GRAPH_ENTITY_SEARCHER


def get_graph_dependency_searcher() -> RepoDependencySearcher:
    global DP_GRAPH_DEPENDENCY_SEARCHER
    return DP_GRAPH_DEPENDENCY_SEARCHER


def get_graph():
    global DP_GRAPH
    assert DP_GRAPH is not None
    return DP_GRAPH

def get_repo_save_dir():
    global REPO_SAVE_DIR
    return REPO_SAVE_DIR


def get_module_name_by_line_num(file_path: str, line_num: int):
    # TODO: 
    # if the given line isn't in a function of a class and the class is large, 
    # find the nearest two member functions and return
    
    entity_searcher = get_graph_entity_searcher()
    dp_searcher = get_graph_dependency_searcher()

    # 尝试标准化文件路径
    normalized_file_path = _normalize_node_id_for_graph(file_path, entity_searcher)

    cur_module = None
    if entity_searcher.has_node(normalized_file_path):
        module_nids, _ = dp_searcher.get_neighbors(normalized_file_path, etype_filter=[EDGE_TYPE_CONTAINS])
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            if module['start_line'] <= line_num <= module['end_line']:
                cur_module = module  # ['node_id']
                break
        if cur_module and cur_module['type'] == NODE_TYPE_CLASS:
            func_nids, _ = dp_searcher.get_neighbors(cur_module['node_id'], etype_filter=[EDGE_TYPE_CONTAINS])
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=True)
            for func in func_ndatas:
                if func['start_line'] <= line_num <= func['end_line']:
                    cur_module = func  # ['node_id']
                    break

    if cur_module: # and cur_module['type'] in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]
        return cur_module
        # module_ndata = entity_searcher.get_node_data([cur_module['node_id']], return_code_content=True)
        # return module_ndata[0]
    return None


def get_code_block_by_line_nums(query_info, context_window=20):
    # file_path: str, line_nums: List[int]
    searcher = get_graph_entity_searcher()
    
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums
    cur_query_results = []

    # 标准化文件路径
    normalized_file_path = _normalize_node_id_for_graph(file_path, searcher)
    
    file_data = searcher.get_node_data([normalized_file_path], return_code_content=False)[0]
    line_intervals = []
    res_modules = []
    # res_code_blocks = None
    for line in line_nums:
        # 首先检查是哪个module的代码
        module_data = get_module_name_by_line_num(normalized_file_path, line)
        
        # 如果不是某个module, 则搜索上下20行
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data['end_line'], line + context_window)
            line_intervals.append((min_line_num, max_line_num))
            
        elif module_data['node_id'] not in res_modules:
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                       nid=module_data['node_id'],
                                       ntype=module_data['type'],
                                       start_line=module_data['start_line'],
                                       end_line=module_data['end_line'],
                                       retrieve_src=f"Retrieved code context including {query_info.term}."
                                       )
            cur_query_results.append(query_result)
            res_modules.append(module_data['node_id'])
            
    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for interval in line_intervals:
            start_line, end_line = interval
            query_result = QueryResult(query_info=query_info, 
                                        format_mode='code_snippet',
                                        nid=normalized_file_path,
                                        file_path=normalized_file_path,
                                        start_line=start_line,
                                        end_line=end_line,
                                        retrieve_src=f"Retrieved code context including {query_info.term}."
                                        )
            cur_query_results.append(query_result)
        # res_code_blocks = line_wrap_content('\n'.join(file_content), line_intervals)

    # return res_code_blocks, res_modules
    return cur_query_results


def parse_node_id(nid: str):
    nfile = nid.split(':')[0]
    nname = nid.split(':')[-1]
    return nfile, nname


def get_embed_model():
    """获取嵌入模型"""
    logger.info("Initializing HuggingFaceEmbedding model: nomic-ai/CodeRankEmbed")
    # For Jina embeddings, it's recommended to trust remote code if prompted,
    # though HuggingFaceEmbedding might handle this.
    # device="cuda" can be added if a GPU is available and preferred.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"device: {device}")
    return HuggingFaceEmbedding(
        model_name="nomic-ai/CodeRankEmbed",
        embed_batch_size=16, # You can adjust this batch size
        device=device,
        trust_remote_code=True
    )
    # return DashScopeEmbedding(
    #     model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    #     text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    #     embed_batch_size=10  # 设置每批处理的文本数量为16
    # )


def load_vector_index(index_dir, repo_path):
    """加载向量索引"""
    try:
        # 创建向量存储和文档存储
        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir=index_dir)
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir=index_dir)

        # 创建索引设置
        settings = IndexSettings(
            embed_model="nomic-ai/CodeRankEmbed",
            dimensions=768
        )

        # 获取embedding模型
        embed_model = get_embed_model()

        # 创建一个空的文件仓库对象，只用于搜索
        file_repo = FileRepository(repo_path)

        # 创建代码索引
        code_index = CodeIndex(
            file_repo=file_repo,
            vector_store=vector_store,
            docstore=docstore,
            embed_model=embed_model,
            settings=settings,
        )

        return code_index
    except Exception as e:
        logger.error(f"加载向量索引失败: {e}")
        return None


def search_entity_in_global_dict(term: str, include_files: Optional[List[str]] = None, prefix_term=None):
    searcher = get_graph_entity_searcher()
    
    # TODO: hard code cases like "class Migration" and "function testing"
    if term.startswith(('class ', 'Class')):
        term = term[len('class '):].strip()
    elif term.startswith(('function ', 'Function ')):
        term = term[len('function '):].strip()
    elif term.startswith(('method ', 'Method ')):
        term = term[len('method '):].strip()
    elif term.startswith('def '):
        term = term[len('def '):].strip()
    
    # TODO: lower case if not find
    # TODO: filename xxx.py as key (also lowercase if not find)
    # global_name_dict = None
    if term in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[term]
    elif term.lower() in searcher.global_name_dict_lowercase:
        term = term.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[term]
    else:
        return None
    
    node_datas = searcher.get_node_data(nids, return_code_content=False)
    found_entities_filter_dict = collections.defaultdict(list)


    # 标准化 include_files 列表以便跨平台匹配
    normalized_include_files = None
    if include_files:
        normalized_include_files = set()
        for f in include_files:
            normalized_include_files.add(f)
            normalized_include_files.add(f.replace('/', '\\'))
            normalized_include_files.add(f.replace('\\', '/'))

    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata['node_id'])

        # 跨平台文件路径匹配
        file_matches = False
        if not include_files:
            file_matches = True
        elif normalized_include_files:
            # 检查原始路径和标准化后的路径
            if nfile in normalized_include_files:
                file_matches = True
            elif nfile.replace('\\', '/') in normalized_include_files:
                file_matches = True
            elif nfile.replace('/', '\\') in normalized_include_files:
                file_matches = True

        if file_matches:
            prefix_terms = []
            # candidite_prefixes = ndata['node_id'].lower().replace('.py', '').replace('/', '.').split('.')
            # 统一使用 / 进行分割处理
            normalized_node_id = ndata['node_id'].replace('\\', '/')
            candidite_prefixes = re.split(r'[./:]', normalized_node_id.lower().replace('.py', ''))[:-1]
            if prefix_term:
                prefix_terms = prefix_term.lower().split('.')
            if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
                found_entities_filter_dict[ndata['type']].append(ndata['node_id'])

    # for ndata in node_datas:
    #     nfile, _ = parse_node_id(ndata['node_id'])
    #     if not include_files or nfile in include_files:
    #         prefix_terms = []
    #         # candidite_prefixes = ndata['node_id'].lower().replace('.py', '').replace('/', '.').split('.')
    #         candidite_prefixes = re.split(r'[./:]', ndata['node_id'].lower().replace('.py', ''))[:-1]
    #         if prefix_term:
    #             prefix_terms = prefix_term.lower().split('.')
    #         if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
    #             found_entities_filter_dict[ndata['type']].append(ndata['node_id'])

    return found_entities_filter_dict


def search_entity(query_info, include_files: List[str] = None):
    term = query_info.term
    searcher = get_graph_entity_searcher()
    # cur_result = ''
    continue_search = True

    cur_query_results = []

    # first: exact match in graph
    # 首先进行实体id索引精确匹配，如果能够直接通过关键词匹配到相应实体则停止搜索
    # 尝试标准化节点ID以处理跨平台路径差异
    normalized_term = _normalize_node_id_for_graph(term, searcher)
    
    # first: exact match in graph
    # 首先进行实体id索引精确匹配，如果能够直接通过关键词匹配到相应实体则停止搜索
    if searcher.has_node(normalized_term):
        continue_search = False
        # 获取节点数据，检查行数，如果行数超过1200行则以preview形式返回，防止超过模型上下文窗口长度
        node_data = searcher.get_node_data([normalized_term], return_code_content=True)[0]
        # if node_data['end_line'] - node_data['start_line'] > 1200:
        if node_data.get('end_line', 0) - node_data.get('start_line', 0) > 1200:
            query_result = QueryResult(query_info=query_info, format_mode='preview',
                                    nid=normalized_term,
                                    ntype=node_data['type'],
                                    start_line=node_data['start_line'],
                                    end_line=node_data['end_line'],
                                    retrieve_src=f"Exact match found for entity name `{term}` (preview mode due to large size)."
            )
        else:
            query_result = QueryResult(query_info=query_info, format_mode='complete', nid=normalized_term,
                                       retrieve_src=f"Exact match found for entity name `{term}`."
                                       )
        cur_query_results.append(query_result)
    
    # TODO: __init__ not exsit
    elif term.endswith('.__init__'):
        nid = term[:-(len('.__init__'))]
        # 尝试标准化节点ID
        normalized_nid = _normalize_node_id_for_graph(nid, searcher)
        if searcher.has_node(normalized_nid):
            continue_search = False
            node_data = searcher.get_node_data([normalized_nid], return_code_content=True)[0]
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                    nid=normalized_nid,
                                    ntype=node_data['type'],
                                    start_line=node_data['start_line'],
                                    end_line=node_data['end_line'],
                                    retrieve_src=f"Exact match found for entity name `{nid}`."
                                    )
            cur_query_results.append(query_result)
    
    # second: search in global name dict
    if continue_search: 
        found_entities_dict = search_entity_in_global_dict(term, include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(term)
        
        use_sub_term = False
        used_term = term
        if not found_entities_dict and '.' in term:
            # for cases: class_name.method_name
            try:
                prefix_term = '.'.join(term.split('.')[:-1]).split()[-1] # incase of 'class '/ 'function '
            except IndexError:
                prefix_term = None
            split_term = term.split('.')[-1].strip()
            used_term = split_term
            found_entities_dict = search_entity_in_global_dict(split_term, include_files, prefix_term)
            if not found_entities_dict:
                found_entities_dict = search_entity_in_global_dict(split_term, prefix_term)
            if not found_entities_dict:
                use_sub_term = True
                found_entities_dict = search_entity_in_global_dict(split_term)
        
        # TODO: split the term and find in global dict
            
        if found_entities_dict:
            for ntype, nids in found_entities_dict.items():
                if not nids: continue
                # if not continue_search: break

                # procee class and function in the same way
                if ntype in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS, NODE_TYPE_FILE]:
                    if len(nids) <= 3:
                        node_datas = searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                                       nid=ndata['node_id'], 
                                                       ntype=ndata['type'],
                                                       start_line=ndata['start_line'],
                                                       end_line=ndata['end_line'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                        # continue_search = False
                    else:
                        node_datas = searcher.get_node_data(nids, return_code_content=False)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                                       nid=ndata['node_id'],
                                                       ntype=ndata['type'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                    if not use_sub_term:
                        continue_search = False
                    else:
                        continue_search = True
                                   
        
    # third: bm25 search (entity + content)
    if continue_search:
        module_nids = []

        # append the file name to keyword?
        # # if not any(symbol in file_path_or_pattern for symbol in ['*','?', '[', ']']):
        # term_with_file = f'{file_path_or_pattern}:{term}'
        # module_nids = bm25_module_retrieve(query=term_with_file, include_files=include_files)

        # search entity by keyword
        module_nids = bm25_module_retrieve(query=term, include_files=include_files)
        if not module_nids:
            module_nids = bm25_module_retrieve(query=term)
            
        if not module_nids:
            # result += f"No entity found using BM25 search. Try to use fuzzy search...\n"
            module_nids = fuzzy_retrieve(term, graph=get_graph(), similarity_top_k=3)

        module_datas = searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        for module in module_datas[:5]:
            if module['type'] in [NODE_TYPE_FILE, NODE_TYPE_DIRECTORY]:
                query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)
            elif showed_module_num < 3:
                showed_module_num += 1
                query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        start_line=module['start_line'],
                                            end_line=module['end_line'],
                                            retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)

    return (cur_query_results, continue_search)


def merge_query_results(query_results):
    priority = ['complete', 'code_snippet', 'preview', 'fold']
    merged_results = {}
    all_query_results: List[QueryResult] = []

    for qr in query_results:
        if qr.format_mode == 'code_snippet':
            all_query_results.append(qr)
        
        elif qr.nid and qr.nid in merged_results:
            # Merge query_info_list
            if qr.query_info_list[0] not in merged_results[qr.nid].query_info_list:
                merged_results[qr.nid].query_info_list.extend(qr.query_info_list)

            # Select the format_mode with the highest priority
            existing_format_mode = merged_results[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_format_mode):
                merged_results[qr.nid].format_mode = qr.format_mode
                merged_results[qr.nid].start_line = qr.start_line
                merged_results[qr.nid].end_line = qr.end_line
                merged_results[qr.nid].retrieve_src = qr.retrieve_src
                
        elif qr.nid:
            merged_results[qr.nid] = qr
    
    all_query_results += list(merged_results.values())
    return all_query_results


def rank_and_aggr_query_results(query_results, fixed_query_info_list):
    query_info_list_dict = {}

    for qr in query_results:
        # Convert the query_info_list to a tuple so it can be used as a dictionary key
        key = tuple(qr.query_info_list)

        if key in query_info_list_dict:
            query_info_list_dict[key].append(qr)
        else:
            query_info_list_dict[key] = [qr]
            
    # for the key: sort by query
    def sorting_key(key):
        # Find the first matching element index from fixed_query_info_list in the key (tuple of query_info_list)
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        # If no match is found, assign a large index to push it to the end
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)
    sorted_query_info_list_dict = {key: query_info_list_dict[key] for key in sorted_keys}
    
    # for the value: sort by format priority
    priority = {'complete': 1, 'code_snippet': 2, 'preview': 3,  'fold': 4}  # Lower value indicates higher priority
    # TODO: merge the same node in 'code_snippet' and 'preview'
    
    organized_dict = {}
    for key, values in sorted_query_info_list_dict.items():
        nested_dict = {priority_key: [] for priority_key in priority.keys()}
        for qr in values:
            # Place the qr in the nested dictionary based on its format_mode
            if qr.format_mode in nested_dict:
                nested_dict[qr.format_mode].append(qr)

        # Only add keys with non-empty lists to keep the result clean
        organized_dict[key] = {k: v for k, v in nested_dict.items() if v}
    
    return organized_dict


def semantic_search_code_snippets(
        query: str,
        file_path_or_pattern: Optional[str] = "**/*.py",
):
    """
    基于query进行相似度检索，在仓库中找到相关的代码块
    Args:
        query: 问题查询
        file_path_or_pattern: 筛选路径

    Returns:
    """
    # TODO: 基于问题描述的稠密检索工具
    files, _, _ = get_current_repo_modules()
    all_file_paths = [file['name'] for file in files]

    result = ""
    # exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths
    # print("所有文件")
    # print(file_path_or_pattern)
    # print(include_files)

    query_info = QueryInfo(term=query)
    # print("query_info")
    # print(query_info)
    query_results = vector_content_retrieve(query_info=query_info, include_files=include_files, top_k=5)
    # print("查询结果")
    # print(query_results)

    result += f'##Searching for `{query}` (Semantic  Search)...\n'
    result += f"### Search Result:\n"

    query_info_list_for_ranking = [query_info]
    merged_results = merge_query_results(query_results)
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list_for_ranking)

    searcher = get_graph_entity_searcher()
    if not searcher:
        logger.info("Graph entity searcher is not available ")
        return "Error: could not format search results due to missing graph searcher"

    has_any_formatted_output = False

    for query_infos_tuple, format_to_results_map in ranked_query_to_results.items():
        current_term_output_segment = ""

        for format_mode, query_results_list in format_to_results_map.items():
            if not query_results_list:
                continue

            formatted_section = ""
            if format_mode == 'fold':
                unique_sources = defaultdict(list)
                for qr in query_results_list:
                    unique_sources[qr.retrieve_src].append(qr)

                for src, qrs_from_src in unique_sources.items():
                    for qr_item in qrs_from_src:
                        formatted_section += qr_item.format_output(searcher)
                    formatted_section += "Source: " + src + "\n"

                if len(query_results_list) > 1:
                    formatted_section += "Hint: Use more detailed query to get the full content of some if needed.\n"
                elif query_results_list:
                    formatted_section += f"Hint: Search `{query_results_list[0].nid}` for the full content if needed.\n"
                formatted_section += "\n"

            elif format_mode == 'complete':
                for qr in query_results_list:
                    formatted_section += qr.format_output(searcher)
                    formatted_section += "\n"
            elif format_mode == 'preview':
                filtered_preview_results = []
                grouped_by_file = defaultdict(list)
                for qr_item in query_results_list:
                    if (qr_item.end_line - qr_item.start_line) < 100:
                        grouped_by_file[qr_item.file_path].append(qr_item)
                    else:
                        filtered_preview_results.append(qr_item)

                for _, results_in_file in grouped_by_file.items():
                    sorted_results = sorted(results_in_file, key=lambda r: (r.start_line, -r.end_line))
                    max_end_line = -1
                    for qr_item_sorted in sorted_results:
                        if qr_item_sorted.end_line > max_end_line:
                            filtered_preview_results.append(qr_item_sorted)
                            max_end_line = max(max_end_line, qr_item_sorted.end_line)
                for qr in filtered_preview_results:
                    formatted_section += qr.format_output(searcher)
                    formatted_section += "\n"

            elif format_mode == 'code_snippet':
                for qr in query_results_list:
                    formatted_section += qr.format_output(searcher)
                    formatted_section += "\n"

            if formatted_section.strip():
                current_term_output_segment += formatted_section
                has_any_formatted_output = True
            # print("格式:")
            # print(format_mode)
            # print(query_results_list)

        result += current_term_output_segment

    if not has_any_formatted_output:
        result += "No locations found after formatting.\n"

    result += '\n\n'

    if not result.strip():
        result += 'No locations found using semantic search.\n\n'
    return result.strip()




def search_code_snippets(
        search_terms: Optional[List[str]] = None,
        line_nums: Optional[List] = None,
        file_path_or_pattern: Optional[str] = "**/*.py",
        retrieval_method: str = "sparse"
) -> str:
    """Searches the codebase to retrieve relevant code snippets based on given queries(terms or line numbers).
    
    This function supports retrieving the complete content of a code entity, 
    searching for code entities such as classes or functions by keywords, or locating specific lines within a file. 
    It also supports filtering searches based on a file path or file pattern.
    
    Note:
    1. If `search_terms` are provided, it searches for code snippets based on each term:
        - If a term is formatted as 'file_path:QualifiedName' (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') ,
          or just 'file_path', the corresponding complete code is retrieved or file content is retrieved.
        - If a term matches a file, class, or function name, matched entities are retrieved.
        - If there is no match with any module name, it attempts to find code snippets that likely contain the term.
        
    2. If `line_nums` is provided, it searches for code snippets at the specified lines within the file defined by 
       `file_path_or_pattern`.

    Args:
        search_terms (Optional[List[str]]): A list of names, keywords, or code snippets to search for within the codebase. 
            Terms can be formatted as 'file_path:QualifiedName' to search for a specific module or entity within a file 
            (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') or as 'file_path' to retrieve the complete content 
            of a file. This can also include potential function names, class names, or general code fragments.

        line_nums (Optional[List[int]]): Specific line numbers to locate code snippets within a specified file. 
            When provided, `file_path_or_pattern` must specify a valid file path.
        
        file_path_or_pattern (Optional[str]): A glob pattern or specific file path used to filter search results 
            to particular files or directories. Defaults to '**/*.py', meaning all Python files are searched by default.
            If `line_nums` are provided, this must specify a specific file path.

    Returns:
        str: The search results, which may include code snippets, matching entities, or complete file content.
        
    
    Example Usage:
        # Search for the full content of a specific file
        result = search_code_snippets(search_terms=['src/my_file.py'])
        
        # Search for a specific function
        result = search_code_snippets(search_terms=['src/my_file.py:MyClass.func_name'])
        
        # Search for specific lines (10 and 15) within a file
        result = search_code_snippets(line_nums=[10, 15], file_path_or_pattern='src/example.py')
        
        # Combined search for a module name and within a specific file pattern
        result = search_code_snippets(search_terms=["MyClass"], file_path_or_pattern="src/**/*.py")
    """
    # print(f"检索方式：{retrieval_method}")
    files, _, _ = get_current_repo_modules()
    all_file_paths = [file['name'] for file in files]

    result = ""
    # exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths

    query_info_list = []
    all_query_results = []
    
    if search_terms:
        # search all terms together
        filter_terms = []
        for term in search_terms:
            if is_test_file(term):
                result += f'No results for test files: `{term}`. Please do not search for any test files.\n\n'
            else:
                filter_terms.append(term)
        
        joint_terms = deepcopy(filter_terms)
        if len(filter_terms) > 1:
            filter_terms.append(' '.join(filter_terms))
        
        for i, term in enumerate(filter_terms):
            term = term.strip().strip('.')
            if not term: continue
                
            query_info = QueryInfo(term=term)
            query_info_list.append(query_info)
            
            cur_query_results = []
            # 新加注释
            # print(f"query_info: \n{query_info}")

            # 根据检索方法选择不同的检索策略
            if retrieval_method.lower() == "dense":
                # print("使用密集检索")
                # 使用密集检索方法（向量检索）
                query_results = vector_content_retrieve(query_info=query_info, include_files=include_files, top_k=5)
                cur_query_results.extend(query_results)
            elif retrieval_method.lower() == "sparse":
                # 默认使用稀疏检索方法（分层检索）
                # search entity
                query_results, continue_search = search_entity(query_info=query_info, include_files=include_files)
                cur_query_results.extend(query_results)

                # bm25检索无法处理特殊字符，导致底层数组越界错误
                import re
                # 检查查询词是否包含特殊字符
                if re.search(r'[^\w\s\.\/\\\-\:]', query_info.term):
                    continue
                
                # search content
                if continue_search:
                    query_results = bm25_content_retrieve(query_info=query_info, include_files=include_files)
                    cur_query_results.extend(query_results)
                    
                elif i != (len(filter_terms)-1):
                    joint_terms[i] = ''
                    filter_terms[-1] = ' '.join([t for t in joint_terms if t.strip()])
                    if filter_terms[-1] in filter_terms[:-1]:
                        filter_terms[-1] = ''
            elif retrieval_method.lower() == "hybrid":
                # 使用混合检索，即先根据search_entity检索，然后根据bm25_content_retrieve以及vector_content_retrieve进行检索，
                # bm25以及密集检索都检索最相近的10个相关实体，然后这些实体进行重排序，将重排序后的结果以及search_entity的检索结果作为混合结果返回
                # 先在全局字典中进行检索
                query_results, continue_search = search_entity(query_info=query_info, include_files=include_files)
                cur_query_results.extend(query_results)

                other_results = []
                query_results_bm = []
                # 使用向量检索
                query_results_vec = vector_content_retrieve(query_info=query_info, include_files=include_files, top_k=5)
                other_results.extend(query_results_vec)
                # 使用bm25检索
                # bm25检索无法处理特殊字符，导致底层数组越界错误
                import re
                # 检查查询词是否包含特殊字符，过滤掉过短的查询词
                if not re.search(r'[^\w\s\.\/\\\-\:]', query_info.term) and len(query_info.term) > 1:
                    query_results_bm = bm25_content_retrieve(query_info=query_info, include_files=include_files, similarity_top_k=10)
                    other_results.extend(query_results_bm)

                    # 判断other_results的长度，如果长度小于10，则不用进行重排序
                    if len(other_results) > 10:
                        # 将bm25的检索结果和向量检索结果进行重排序，具体来使用marjority算法，这一步只统计前10的文件
                        # 将query_results_bm和query_results_vec列表合并成一个列表，然后使用majority算法进行重排序
                        query_formaj_results = query_results_vec + query_results_bm
                        loc_counts = collections.Counter(query_formaj_results)
                        loc_weights = collections.defaultdict(float)
                        for loc, count in loc_counts.items():
                            nid = loc.nid
                            # 这个nid是实体名称，可能是文件名称也可能是类或者函数名称，需要从中提取出文件名称
                            fn = nid.split(":")[0]
                            loc_weights[fn] = count
                        ranked_loc_weights = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)
                        # 只取前10个文件
                        ranked_locs = [file for file, _ in ranked_loc_weights[:5]]

                        # 从other_results中筛选出文件名称包含在ranked_locs中的实体信息
                        # res_other = [result for result in other_results if result.nid.split(":")[0] in ranked_locs]

                        # 修改为：
                        def _is_path_match(path1: str, path2: str) -> bool:
                            """检查两个路径是否匹配（忽略分隔符差异）"""
                            if not path1 or not path2:
                                return False
                            return path1.replace('\\', '/') == path2.replace('\\', '/')

                        # 然后修改过滤逻辑：
                        res_other = []
                        for result in other_results:
                            result_file = result.nid.split(":")[0]
                            for ranked_file in ranked_locs:
                                if _is_path_match(result_file, ranked_file):
                                    res_other.append(result)
                                    break

                        # 对res_other根据nid进行去重，防止提供过多无关数据
                        filter_res_other = []
                        for result in res_other:
                            if result.nid not in [res.nid for res in filter_res_other]:
                                filter_res_other.append(result)
                        other_results = filter_res_other
                cur_query_results.extend(other_results)


            # search entity
            # query_results, continue_search = search_entity(query_info=query_info, include_files=include_files)
            # cur_query_results.extend(query_results)
            
            # # search content
            # if continue_search:
            #     query_results = bm25_content_retrieve(query_info=query_info, include_files=include_files)
            #     cur_query_results.extend(query_results)
                
            # elif i != (len(filter_terms)-1):
            #     joint_terms[i] = ''
            #     filter_terms[-1] = ' '.join([t for t in joint_terms if t.strip()])
            #     if filter_terms[-1] in filter_terms[:-1]:
            #         filter_terms[-1] = ''
                
            all_query_results.extend(cur_query_results)

    def _is_file_in_list(file_path: str, file_list: List[str]) -> bool:
        """检查文件路径是否在列表中（支持跨平台匹配）"""
        if not file_path or not file_list:
            return False
        if file_path in file_list:
            return True
        # 尝试不同的分隔符
        normalized_path = file_path.replace('\\', '/')
        if normalized_path in file_list:
            return True
        normalized_path = file_path.replace('/', '\\')
        if normalized_path in file_list:
            return True
        # 检查列表中是否有匹配的路径
        for f in file_list:
            if f.replace('\\', '/') == file_path.replace('\\', '/'):
                return True
        return False

    # if file_path_or_pattern in all_file_paths and line_nums:
    if _is_file_in_list(file_path_or_pattern, all_file_paths) and line_nums:
        if isinstance(line_nums, int):
            line_nums = [line_nums]
        file_path = file_path_or_pattern
        # 找到实际匹配的路径
        for f in all_file_paths:
            if f.replace('\\', '/') == file_path.replace('\\', '/'):
                file_path = f
                break
        term = file_path + ':line ' + ', '.join([str(line) for line in line_nums])
        # result += f"Search `line(s) {line_nums}` in file `{file_path}` ...\n"
        query_info = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)
        
        # Search for codes based on file name and line number
        query_results = get_code_block_by_line_nums(query_info)
        all_query_results.extend(query_results)
    
    
    merged_results = merge_query_results(all_query_results)
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list)
    # 遍历输出rranked_query_to_results字典中的内容，用于调试
    # print("zengcong")
    # for query_infos, format_to_results in ranked_query_to_results.items():
    #     print(f"query_infos:\n{query_infos}")
    #     # format_to_results是QueryResult，其重写了__str__方法
    #     for format_mode, query_results in format_to_results.items():
    #         print(f"format_mode:\n{format_mode}")
    #         for query_result in query_results:
    #             print(query_result)

    
    # format output
    # format_mode: 'complete', 'preview', 'code_snippet', 'fold': 4
    searcher = get_graph_entity_searcher()
    # print(f"ranked_query_to_results:\n{ranked_query_to_results}")
    for query_infos, format_to_results in ranked_query_to_results.items():
        term_desc = ', '.join([f'"{query.term}"' for query in query_infos])
        result += f'##Searching for term {term_desc}...\n'
        result += f'### Search Result:\n'
        cur_result = ''
        for format_mode, query_results in format_to_results.items():
            if format_mode == 'fold':
                cur_retrieve_src = ''
                for qr in query_results:
                    if not cur_retrieve_src:
                        cur_retrieve_src = qr.retrieve_src
                        
                    if cur_retrieve_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_retrieve_src + '\n\n'
                        cur_retrieve_src = qr.retrieve_src
                        
                    cur_result += qr.format_output(searcher)
                    
                cur_result += "Source: " + cur_retrieve_src + '\n'
                if len(query_results) > 1:
                    cur_result += 'Hint: Use more detailed query to get the full content of some if needed.\n'
                else:
                    cur_result += f'Hint: Search `{query_results[0].nid}` for the full content if needed.\n'
                cur_result += '\n'
                
            elif format_mode == 'complete':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'

            elif format_mode == 'preview':
                # Remove the small modules, leaving only the large ones
                filtered_results = []
                grouped_by_file = defaultdict(list)
                for qr in query_results:
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)
                
                for file_path, results in grouped_by_file.items():
                    # Sort by start_line and then by end_line in descending order
                    sorted_results = sorted(results, key=lambda qr: (qr.start_line, -qr.end_line))

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
                        if qr.end_line > max_end_line:
                            filtered_results.append(qr)
                            max_end_line = max(max_end_line, qr.end_line)
                
                # filtered_results = query_results
                for qr in filtered_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'
            
            elif format_mode == 'code_snippet':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'
            
        cur_result += '\n\n'
        
        if cur_result.strip():
            result += cur_result
        else:
            result += 'No locations found.\n\n'
        
    return result.strip()


def get_entity_contents(entity_names: List[str]):
    searcher = get_graph_entity_searcher()
    
    result = ''
    for name in entity_names:
        name = name.strip().strip('.')
        if not name: continue
        
        result += f'##Searching for entity `{name}`...\n'
        result += f'### Search Result:\n'
        query_info = QueryInfo(term=name)

        # 尝试标准化节点ID以处理跨平台路径差异
        normalized_name = _normalize_node_id_for_graph(name, searcher)
        
        if searcher.has_node(normalized_name):
            query_result = QueryResult(query_info=query_info, format_mode='complete', nid=normalized_name,
                                    retrieve_src=f"Exact match found for entity name `{name}`."
                                    )
            result += query_result.format_output(searcher)
            result += '\n\n'
        else:
            result += 'Invalid name. \nHint: Valid entity name should be formatted as "file_path:QualifiedName" or just "file_path", or the entity you are searching for does not exist.'
            result += '\n\n'
    return result.strip()

def _detect_path_separator(file_paths: List[str]) -> str:
    """
    检测文件路径列表中使用的路径分隔符。

    Args:
        file_paths: 文件路径列表

    Returns:
        '/' 或 '\\' 表示检测到的分隔符，默认返回 '/'
    """
    if not file_paths:
        return '/'

    # 统计两种分隔符的出现次数
    forward_slash_count = 0
    back_slash_count = 0

    for path in file_paths[:20]:  # 只检查前20个路径以提高效率
        forward_slash_count += path.count('/')
        back_slash_count += path.count('\\')

    # 返回出现次数更多的分隔符
    if back_slash_count > forward_slash_count:
        return '\\'
    return '/'


def _normalize_file_path(file_path: str, include_files: List[str] = None, repo_name: str = None) -> str:
    """
    将向量索引中的文件路径标准化为图中使用的相对路径格式。

    处理逻辑：
    1. 检测 include_files 中使用的路径分隔符
    2. 统一 file_path 的分隔符
    3. 去除仓库前缀（如 playground/repo_name/）
    4. 尝试匹配 include_files 中的路径

    Args:
        file_path: 向量索引中的文件路径
        include_files: 图中的文件路径列表
        repo_name: 仓库名称（格式如 'ultralytics__ultralytics'）

    Returns:
        标准化后的文件路径
    """
    if not file_path:
        return file_path

    # 如果已经在 include_files 中，直接返回
    if include_files and file_path in include_files:
        return file_path

    # 检测 include_files 中使用的分隔符
    target_sep = _detect_path_separator(include_files) if include_files else '/'
    source_sep = '\\' if '\\' in file_path else '/'

    # 统一使用正斜杠进行处理
    normalized_path = file_path.replace('\\', '/')

    # 去除仓库前缀
    # 常见模式: playground/repo_name/actual/path.py 或 playground/owner__repo/actual/path.py
    prefixes_to_remove = []

    if repo_name:
        prefixes_to_remove.extend([
            f"playground/{repo_name}/",
            f"playground/repo/{repo_name}/",
        ])

    # 通用前缀模式
    prefixes_to_remove.extend([
        "playground/",
    ])

    # 尝试去除前缀
    for prefix in prefixes_to_remove:
        if normalized_path.startswith(prefix):
            # 对于 "playground/" 这种通用前缀，需要额外去除仓库名目录
            if prefix == "playground/":
                # playground/repo_name/actual/path.py -> actual/path.py
                parts = normalized_path[len(prefix):].split('/', 1)
                if len(parts) > 1:
                    normalized_path = parts[1]
                break
            else:
                normalized_path = normalized_path[len(prefix):]
                break

    # 尝试匹配 include_files 中的路径
    if include_files:
        # 将 include_files 中的路径也统一为正斜杠格式进行比较
        normalized_include_files = {p.replace('\\', '/'): p for p in include_files}

        # 直接匹配
        if normalized_path in normalized_include_files:
            return normalized_include_files[normalized_path]

        # 后缀匹配：检查 normalized_path 是否以 include_files 中某个路径结尾
        for norm_inc_path, orig_inc_path in normalized_include_files.items():
            if normalized_path.endswith('/' + norm_inc_path) or normalized_path == norm_inc_path:
                return orig_inc_path
            # 反向检查：include_files 中的路径是否是 normalized_path 的后缀
            if norm_inc_path.endswith('/' + normalized_path) or norm_inc_path == normalized_path:
                return orig_inc_path

    # 如果没有匹配到，根据目标分隔符转换路径
    if target_sep == '\\':
        return normalized_path.replace('/', '\\')
    return normalized_path


def vector_content_retrieve(query_info, include_files=None, top_k = 10):
    """
    使用向量搜索来查找与给定查询最相关的代码片段。

    Args:
        query_info (QueryInfo): 查询信息对象，包含要搜索的术语和其他相关信息。
        include_files (List[str]): 可选参数，指定需要包含的文件路径列表。

    Returns:
        List[Dict]: 与查询最相关的代码片段列表。
    """
    try:
        # 获取问题描述作为查询
        query = query_info.term

        # 获取当前实例ID和仓库路径
        global CURRENT_ISSUE_ID, REPO_SAVE_DIR

        # 设置索引目录和仓库路径
        index_dir = os.path.join(DENSE_INDEX_DIR, CURRENT_ISSUE_ID)
        
        # 尝试获取共享仓库路径
        repo_path = None
        if CURRENT_INSTANCE and 'repo' in CURRENT_INSTANCE:
            # 处理共享仓库路径: playground/owner__repo
            repo_name = CURRENT_INSTANCE['repo'].replace('/', '__')
            shared_repo_path = os.path.join("playground", "repo", repo_name)
            
            if os.path.exists(shared_repo_path):
                repo_path = shared_repo_path
                # 检查并切换 commit
                if 'base_commit' in CURRENT_INSTANCE:
                    base_commit = CURRENT_INSTANCE['base_commit']
                    try:
                        # 获取当前 commit
                        current_commit = subprocess.check_output(
                            ['git', 'rev-parse', 'HEAD'], 
                            cwd=repo_path, 
                            text=True
                        ).strip()
                        
                        # 如果 commit 不匹配，则切换
                        if current_commit != base_commit:
                            logger.info(f"Switching repo {repo_name} from {current_commit[:8]} to {base_commit[:8]}")
                            # 强制切换，先 clean 再 checkout
                            subprocess.run(['git', 'clean', '-fd'], cwd=repo_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            subprocess.run(['git', 'reset', '--hard'], cwd=repo_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            subprocess.run(['git', 'checkout', '-f', base_commit], cwd=repo_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as e:
                        logger.warning(f"Failed to check/switch commit for {repo_path}: {e}")
        
        # 如果没有找到共享仓库或发生错误，回退到旧路径逻辑
        if not repo_path:
            # repo_path = REPO_SAVE_DIR
            # repo_path这里先固定仓库所在目录为playground
            repo_path = os.path.join("playground", "repo",  CURRENT_ISSUE_ID)

        # 检查索引目录是否存在
        if not os.path.exists(index_dir):
            logger.warning(f"索引目录不存在: {index_dir}")
            raise  Exception("索引目录不存在")
            return None

        # 加载向量索引
        code_index = load_vector_index(index_dir, repo_path)
        if not code_index:
            logger.warning("无法加载向量索引")
            return []

        # 搜索相关代码
        file_pattern = None
        if include_files and len(include_files) > 0:
            file_pattern = include_files[0] if len(include_files) == 1 else None

        # 使用向量搜索
        results = code_index._vector_search(
            query=query,
            exact_query_match=False,
            category='implementation',  # 只搜索实现代码，不包括测试代码
            file_pattern=file_pattern,
            similarity_top_k=top_k,
            similarity_threshold=0.4,
        )

        # 提取结果并转换为QueryResult对象
        query_results = []
        searcher = get_graph_entity_searcher()
        
        for snippet in results:
            # 从 docstore 获取完整的节点信息
            node = code_index._docstore.get_document(snippet.id)
            metadata = node.metadata

            # 获取原始文件路径
            raw_file_path = metadata.get('file_path', '')

            # 将向量索引中的路径转换为图中使用的格式
            file = _normalize_file_path(raw_file_path, include_files, repo_name)

            # 调试日志
            if raw_file_path != file:
                logger.debug(f"路径标准化: '{raw_file_path}' -> '{file}'")


            # 检查是否在包含文件列表中
            if not include_files or file in include_files:
                # 检查是否为导入代码或文档字符串
                if 'span_ids' in metadata:
                    # 如果所有span_id都是docstring、imports或comments，作为代码片段处理
                    if all([span_id in ['docstring', 'imports', 'comments'] for span_id in metadata['span_ids']]):
                        # query_result = QueryResult(
                        #     query_info=query_info,
                        #     format_mode='code_snippet',
                        #     nid=file,
                        #     file_path=file,
                        #     start_line=metadata.get('start_line', 0),
                        #     end_line=metadata.get('end_line', 0),
                        #     retrieve_src=f"Retrieved code content using vector search."
                        # )
                        # query_results.append(query_result)

                        # 这里先把这种导入语句定位结果去除
                        continue

                    # 如果部分span_id是docstring、imports或comments
                    elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in metadata['span_ids']]):
                        nids = []
                        for span_id in metadata['span_ids']:
                            nid = f'{file}:{span_id}'
                            normalized_nid = _normalize_node_id_for_graph(nid, searcher)
                            if searcher.has_node(normalized_nid):
                                nids.append(normalized_nid)
                            # if searcher.has_node(nid):
                            #     nids.append(nid)

                        node_datas = searcher.get_node_data(nids, return_code_content=True)
                        sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                        sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]

                        message = ''
                        if sorted_nids:
                            if sorted_ndatas[0]['start_line'] < metadata.get('start_line', 0):
                                nid = sorted_ndatas[0]['node_id']
                                ntype = sorted_ndatas[0]['type']
                                message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                            if sorted_ndatas[-1]['end_line'] > metadata.get('end_line', 0):
                                nid = sorted_ndatas[-1]['node_id']
                                ntype = sorted_ndatas[-1]['type']
                                message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                            if message.strip():
                                message = "Hint: \n"+ message

                        nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                        desc = f"Found {nids_str}."
                        query_result = QueryResult(
                            query_info=query_info,
                            format_mode='code_snippet',
                            nid=file,
                            file_path=file,
                            start_line=metadata.get('start_line', 0),
                            end_line=metadata.get('end_line', 0),
                            desc=desc,
                            message=message,
                            retrieve_src=f"Retrieved code content using vector search."
                        )

                        query_results.append(query_result)
                    else:
                        # 处理其他类型的span_id
                        for span_id in metadata['span_ids']:
                            nid = f'{file}:{span_id}'
                            normalized_nid = _normalize_node_id_for_graph(nid, searcher)
                            if searcher.has_node(normalized_nid):
                                ndata = searcher.get_node_data([normalized_nid], return_code_content=True)[0]
                                query_result = QueryResult(
                                    query_info=query_info,
                                    format_mode='preview',
                                    nid=ndata['node_id'],
                                    ntype=ndata['type'],
                                    start_line=ndata['start_line'],
                                    end_line=ndata['end_line'],
                                    retrieve_src=f"Retrieved code content using vector search."
                                )
                                query_results.append(query_result)
                else:
                    # 处理没有span_ids的情况
                    query_result = QueryResult(
                        query_info=query_info,
                        format_mode='code_snippet',
                        nid=file,
                        file_path=file,
                        start_line=metadata.get('start_line', 0),
                        end_line=metadata.get('end_line', 0),
                        retrieve_src=f"Retrieved code content using vector search."
                    )
                    query_results.append(query_result)

        # 限制返回结果数量
        cur_query_results = query_results[:top_k]
        return cur_query_results

    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        return []



def build_code_vector_retriever_from_repo(
        repo_path: str,
        similarity_top_k: int = 10,
        min_chunk_size: int = 100,
        chunk_size: int = 500,
        max_chunk_size: int = 2000,
        hard_token_limit: int = 2000,
        max_chunks: int = 200,
        persist_path: Optional[str] = None,
        show_progress: bool = False,
):
    """构建代码向量检索器"""
    try:
        # 定义文件元数据提取函数
        def file_metadata_func(file_path: str) -> Dict:
            relative_path = os.path.relpath(file_path, repo_path)
            test_patterns = [
                '**/test/**',
                '**/tests/**',
                '**/test_*.py',
                '**/*_test.py',
            ]
            category = (
                'test'
                if any(fnmatch.fnmatch(relative_path, pattern) for pattern in test_patterns)
                else 'implementation'
            )
            return {
                'file_path': relative_path,
                'category': category,
            }

        # 创建文件仓库
        file_repo = FileRepository(repo_path)

        # 创建索引设置
        settings = IndexSettings(
            embed_model="text-embedding-v3",
            dimensions=1024
        )

        # 获取embedding模型
        from vector_demo import get_embed_model
        embed_model = get_embed_model()

        # 创建向量存储和文档存储
        import faiss
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

        # 构建索引
        code_index.build_index(
            show_progress=show_progress,
            file_metadata_func=file_metadata_func,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            max_chunk_size=max_chunk_size,
            hard_token_limit=hard_token_limit,
            max_chunks=max_chunks,
        )

        # 持久化索引
        if persist_path:
            code_index.persist(persist_dir=persist_path)

        return code_index
    except Exception as e:
        logger.error(f"构建向量索引失败: {e}")
        return None
    

def bm25_module_retrieve(
        query: str,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        search_scope: str = 'all',
        similarity_top_k: int = 10,
        # sort_by_type = False
):
    retriever = build_module_retriever(entity_searcher=get_graph_entity_searcher(),
                                       search_scope=search_scope,
                                       similarity_top_k=similarity_top_k)
    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        logging.warning(f'{e}. Probably because the query `{query}` is too short.')
        return []

    # 标准化 include_files 列表以便跨平台匹配
    normalized_include_files = None
    if include_files:
        normalized_include_files = set()
        for f in include_files:
            normalized_include_files.add(f)
            normalized_include_files.add(f.replace('/', '\\'))
            normalized_include_files.add(f.replace('\\', '/'))

    filter_nodes = []
    all_nodes = []
    for node in retrieved_nodes:
        if node.score <= 0:
            continue

        node_file = node.text.split(':')[0]

        # 跨平台文件匹配
        file_matches = False
        if not include_files:
            file_matches = True
        elif normalized_include_files:
            if node_file in normalized_include_files:
                file_matches = True
            elif node_file.replace('\\', '/') in normalized_include_files:
                file_matches = True
            elif node_file.replace('/', '\\') in normalized_include_files:
                file_matches = True

        if file_matches:
            filter_nodes.append(node.text)
        all_nodes.append(node.text)
    # for node in retrieved_nodes:
    #     if node.score <= 0:
    #         continue
    #     if not include_files or node.text.split(':')[0] in include_files:
    #         filter_nodes.append(node.text)
    #     all_nodes.append(node.text)

    if filter_nodes:
        return filter_nodes
    else:
        return all_nodes


def bm25_content_retrieve(
        query_info: QueryInfo,
        include_files: Optional[List[str]] = None,
        similarity_top_k: int = 10
) -> str:
    """..."""

    instance = get_current_issue_data()
    query = query_info.term

    persist_path = os.path.join(BM25_INDEX_DIR, instance["instance_id"])
    if os.path.exists(f'{persist_path}/corpus.jsonl'):
        retriever = load_retriever(persist_path)
    else:
        repo_playground = get_repo_save_dir()
        repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_playground, dataset=None, split=None)
        absolute_repo_dir = os.path.abspath(repo_dir)
        retriever = build_code_retriever(absolute_repo_dir, persist_path=persist_path,
                                         similarity_top_k=similarity_top_k)

    # 标准化 include_files 列表以便跨平台匹配
    normalized_include_files = None
    if include_files:
        normalized_include_files = set()
        for f in include_files:
            normalized_include_files.add(f)
            normalized_include_files.add(f.replace('/', '\\'))
            normalized_include_files.add(f.replace('\\', '/'))

    cur_query_results = []
    retrieved_nodes = retriever.retrieve(query)
    searcher = get_graph_entity_searcher()

    for node in retrieved_nodes:
        file = node.metadata['file_path']

        # 跨平台文件匹配
        file_matches = False
        if not include_files:
            file_matches = True
        elif normalized_include_files:
            if file in normalized_include_files:
                file_matches = True
            elif file.replace('\\', '/') in normalized_include_files:
                file_matches = True
            elif file.replace('/', '\\') in normalized_include_files:
                file_matches = True

        if file_matches:
            if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                query_result = QueryResult(query_info=query_info,
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                cur_query_results.append(query_result)

            elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                nids = []
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    # 标准化节点ID
                    normalized_nid = _normalize_node_id_for_graph(nid, searcher)
                    if searcher.has_node(normalized_nid):
                        nids.append(normalized_nid)

                node_datas = searcher.get_node_data(nids, return_code_content=True)
                sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]

                message = ''
                if sorted_nids:
                    if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
                        nid = sorted_ndatas[0]['node_id']
                        ntype = sorted_ndatas[0]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
                        nid = sorted_ndatas[-1]['node_id']
                        ntype = sorted_ndatas[-1]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if message.strip():
                        message = "Hint: \n"+ message

                nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                desc = f"Found {nids_str}."
                query_result = QueryResult(query_info=query_info,
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           desc=desc,
                                           message=message,
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )

                cur_query_results.append(query_result)
            else:
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    # 标准化节点ID
                    normalized_nid = _normalize_node_id_for_graph(nid, searcher)
                    if searcher.has_node(normalized_nid):
                        ndata = searcher.get_node_data([normalized_nid], return_code_content=True)[0]
                        query_result = QueryResult(query_info=query_info, format_mode='preview',
                                                   nid=ndata['node_id'],
                                                   ntype=ndata['type'],
                                                   start_line=ndata['start_line'],
                                                   end_line=ndata['end_line'],
                                                   retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                                   )
                        cur_query_results.append(query_result)
                    else:
                        continue

    cur_query_results = cur_query_results[:5]
    return cur_query_results

# def bm25_content_retrieve(
#         query_info: QueryInfo,
#         # query: str,
#         include_files: Optional[List[str]] = None,
#         # file_pattern: Optional[str] = None,
#         similarity_top_k: int = 10
# ) -> str:
#     """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.
#
#     Args:
#         query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
#         class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
#         function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
#         file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
#         similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.
#
#     Returns:
#         str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
#     """
#
#     instance = get_current_issue_data()
#     query = query_info.term
#
#     persist_path = os.path.join(BM25_INDEX_DIR, instance["instance_id"])
#     if os.path.exists(f'{persist_path}/corpus.jsonl'):
#         # TODO: if similairy_top_k > cache's setting, then regenerate
#         retriever = load_retriever(persist_path)
#     else:
#         repo_playground = get_repo_save_dir()
#         repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_playground, dataset=None, split=None)
#         absolute_repo_dir = os.path.abspath(repo_dir)
#         retriever = build_code_retriever(absolute_repo_dir, persist_path=persist_path,
#                                          similarity_top_k=similarity_top_k)
#
#     # 标准化 include_files 列表以便跨平台匹配
#     normalized_include_files = None
#     if include_files:
#         normalized_include_files = set()
#         for f in include_files:
#             normalized_include_files.add(f)
#             normalized_include_files.add(f.replace('/', '\\'))
#             normalized_include_files.add(f.replace('\\', '/'))
#
#
#     # similarity: {score}
#     cur_query_results = []
#     retrieved_nodes = retriever.retrieve(query)
#     for node in retrieved_nodes:
#         file = node.metadata['file_path']
#         # print(node.metadata)
#         if not include_files or file in include_files:
#             # drop the import code
#             # if len(node.metadata['span_ids']) == 1 and node.metadata['span_ids'][0] == 'imports':
#             #     continue
#             if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
#                 # TODO: drop ?
#                 query_result = QueryResult(query_info=query_info,
#                                            format_mode='code_snippet',
#                                            nid=node.metadata['file_path'],
#                                            file_path=node.metadata['file_path'],
#                                            start_line=node.metadata['start_line'],
#                                            end_line=node.metadata['end_line'],
#                                            retrieve_src=f"Retrieved code content using keyword search (bm25)."
#                                            )
#                 cur_query_results.append(query_result)
#
#             elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
#                 nids = []
#                 for span_id in node.metadata['span_ids']:
#                     nid = f'{file}:{span_id}'
#                     searcher = get_graph_entity_searcher()
#                     if searcher.has_node(nid):
#                         nids.append(nid)
#                     # TODO: warning if not find
#
#                 node_datas = searcher.get_node_data(nids, return_code_content=True)
#                 sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
#                 sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]
#
#                 message = ''
#                 if sorted_nids:
#                     if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
#                         nid = sorted_ndatas[0]['node_id']
#                         ntype = sorted_ndatas[0]['type']
#                         # The code for {ntype} {nid} is incomplete; search {nid} for the full content if needed.
#                         message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
#                     if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
#                         nid = sorted_ndatas[-1]['node_id']
#                         ntype = sorted_ndatas[-1]['type']
#                         message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
#                     if message.strip():
#                         message = "Hint: \n"+ message
#
#                 nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
#                 desc = f"Found {nids_str}."
#                 query_result = QueryResult(query_info=query_info,
#                                            format_mode='code_snippet',
#                                            nid=node.metadata['file_path'],
#                                            file_path=node.metadata['file_path'],
#                                            start_line=node.metadata['start_line'],
#                                            end_line=node.metadata['end_line'],
#                                            desc=desc,
#                                            message=message,
#                                            retrieve_src=f"Retrieved code content using keyword search (bm25)."
#                                            )
#
#                 cur_query_results.append(query_result)
#             else:
#                 for span_id in node.metadata['span_ids']:
#                     nid = f'{file}:{span_id}'
#                     searcher = get_graph_entity_searcher()
#                     if searcher.has_node(nid):
#                         ndata = searcher.get_node_data([nid], return_code_content=True)[0]
#                         query_result = QueryResult(query_info=query_info, format_mode='preview',
#                                                    nid=ndata['node_id'],
#                                                    ntype=ndata['type'],
#                                                    start_line=ndata['start_line'],
#                                                    end_line=ndata['end_line'],
#                                                    retrieve_src=f"Retrieved code content using keyword search (bm25)."
#                                                    )
#                         cur_query_results.append(query_result)
#                     else:
#                         continue
#
#         cur_query_results = cur_query_results[:5]
#         return cur_query_results


def _validate_graph_explorer_inputs(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 1,
        node_type_filter: Optional[List[str]] = None,
        edge_type_filter: Optional[List[str]] = None,
):
    """evaluate input arguments
    """

    # assert len(invalid_entities) == 0, (
    #     f"Invalid value for `start_entities`: entities {invalid_entities} are not in the repository graph."
    # )
    assert direction in ['downstream', 'upstream', 'both'], (
        "Invalid value for `direction`: Expected one of 'downstream', 'upstream', and 'both'. "
        f"Received: '{direction}'."
    )
    assert traversal_depth == -1 or traversal_depth >= 0, (
        "Invalid value for `traversal_depth`: It must be either -1 or a non-negative integer (>= 0). "
        f"Received: {traversal_depth}."
    )
    if isinstance(node_type_filter, list):
        invalid_ntypes = []
        for ntype in node_type_filter:
            if ntype not in VALID_NODE_TYPES:
                invalid_ntypes.append(ntype)
        assert len(
            invalid_ntypes) == 0, f"Invalid node types {invalid_ntypes} in node_type_filter. Expected node type in {VALID_NODE_TYPES}"
    if isinstance(edge_type_filter, list):
        invalid_etypes = []
        for etype in edge_type_filter:
            if etype not in VALID_EDGE_TYPES:
                invalid_etypes.append(etype)
        assert len(
            invalid_etypes) == 0, f"Invalid edge types {invalid_etypes} in edge_type_filter. Expected edge type in {VALID_EDGE_TYPES}"

    graph = get_graph()
    entity_searcher = get_graph_entity_searcher()

    hints = ''
    valid_entities = []
    for i, root in enumerate(start_entities):
        # process node name
        if root != '/':
            root = root.strip('/')
        if root.endswith('.__init__'):
            root = root[:-(len('.__init__'))]

        # validate node name
        if root not in graph:
            # Try switching separators to handle cross-platform mismatch
            if root.replace('\\', '/') in graph:
                root = root.replace('\\', '/')
            elif root.replace('/', '\\') in graph:
                root = root.replace('/', '\\')

        if root not in graph:
            # search with bm25
            module_nids = bm25_module_retrieve(query=root)
            module_datas = entity_searcher.get_node_data(module_nids, return_code_content=False)
            if len(module_datas) > 0:
                hints += f'The entity name `{root}` is invalid. Based on your input, here are some candidate entities you might be referring to:\n'
                for module in module_datas[:5]:
                    ntype = module['type']
                    nid = module['node_id']
                    hints += f'{ntype}: `{nid}`\n'
                hints += "Source: Retrieved entity using keyword search (bm25).\n\n"
            else:
                hints += f'The entity name `{root}` is invalid. There are no possible candidate entities in record.\n'
        elif is_test_file(root):
            hints += f'No results for the test entity: `{root}`. Please do not include any test entities.\n\n'
        else:
            valid_entities.append(root)

    return valid_entities, hints


def explore_graph_structure(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 1,
        entity_type_filter: Optional[List[str]] = None,
        dependency_type_filter: Optional[List[str]] = None,
        # input_node_ids: List[str],
        # direction: str = 'forward',
        # traverse_hop: int = 1,
        # node_type_filter: Optional[List[str]] = None,
        # edge_type_filter: Optional[List[str]] = None,
        # return_code_content: bool = False,
):
    """
    Args:
        start_entities:
        direction:
        traversal_depth:
        entity_type_filter:
        dependency_type_filter:

    Returns:
    """
    start_entities, hints = _validate_graph_explorer_inputs(start_entities, direction, traversal_depth,
                                            entity_type_filter, dependency_type_filter)
    G = get_graph()

    rtn_str = traverse_graph_structure(G, start_entities, direction, traversal_depth,
                                       entity_type_filter, dependency_type_filter)

    if hints.strip():
        rtn_str += "\n\n" + hints
    return rtn_str.strip()


def explore_tree_structure(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 2,
        entity_type_filter: Optional[List[str]] = None,
        dependency_type_filter: Optional[List[str]] = None,
):
    """Analyzes and displays the dependency structure around specified entities in a code graph.

    This function searches and presents relationships and dependencies for the specified entities (such as classes, functions, files, or directories) in a code graph.
    It explores how the input entities relate to others, using defined types of dependencies, including 'contains', 'imports', 'invokes' and 'inherits'.
    The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional limits on traversal depth and filters for entity and dependency types.

    Example Usage:
    1. Exploring Outward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_a.py:ClassA'],
            direction='downstream',
            traversal_depth=2,
            entity_type_filter=['class', 'function'],
            dependency_type_filter=['invokes', 'imports']
        )
        ```
        This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

    2. Exploring Inward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_b.py:FunctionY'],
            direction='upstream',
            traversal_depth=-1
        )
        ```
        This finds all entities that depend on `FunctionY` without restricting the traversal depth.

    Notes:
    * Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
    * Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.
    * Graph Context: The function operates on a pre-built code graph containing entities (e.g., files, classes and functions) and dependencies representing their interactions and relationships.

    Parameters:
    ----------
    start_entities : list[str]
        List of entities (e.g., class, function, file, or directory paths) to begin the search from.
        - Entities representing classes or functions must be formatted as "file_path:QualifiedName"
          (e.g., `interface/C.py:C.method_a.inner_func`).
        - For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).

    direction : str, optional
        Direction of traversal in the code graph; allowed options are:
        - 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).
        - 'downstream': Traversal to explore the effects or interactions of the specified entities on others
          (how others depend on them).
        - 'both': Traversal in both directions.
        Default is 'downstream'.

    traversal_depth : int, optional
        Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).
        Must be either `-1` or a non-negative integer (≥ 0).
        Default is 2.

    entity_type_filter : list[str], optional
        List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal.
        If None, all entity types are included.
        Default is None.

    dependency_type_filter : list[str], optional
        List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal.
        If None, all dependency types are included.
        Default is None.

    Returns:
    -------
    result : object
        An object representing the traversal results, which includes discovered entities and their dependencies.
    """
    start_entities, hints = _validate_graph_explorer_inputs(start_entities, direction, traversal_depth,
                                                            entity_type_filter, dependency_type_filter)
    G = get_graph()

    # return_json = True
    return_json = False
    if return_json:
        rtns = {node: traverse_json_structure(G, node, direction, traversal_depth, entity_type_filter,
                                              dependency_type_filter)
                for node in start_entities}
        rtn_str = json.dumps(rtns)
    else:
        rtns = [traverse_tree_structure(G, node, direction, traversal_depth, entity_type_filter,
                                        dependency_type_filter)
                for node in start_entities]
        rtn_str = "\n\n".join(rtns)
        
    if hints.strip():
        rtn_str += "\n\n" + hints
    return rtn_str.strip()


__all__ = [
    # 'get_entity_contents',
    'search_code_snippets',
    'explore_graph_structure',
    'explore_tree_structure',
    'semantic_search_code_snippets'
]

# if __name__ == '__main__':
#     # 测试explore_tree_structure
#     # 先从swe-bench-lite中获取astropy__astropy-6938的数据，从huggingface获取
#     datas = load_dataset("czlll/SWE-bench_Lite", split="test")
#     # 设置日志
#     logger = logging.getLogger()
#     logger.setLevel(logging.getLevelName("INFO"))
#     for bug in datas:
#         instance_id = bug["instance_id"]
#         if instance_id == "astropy__astropy-6938":
#             set_current_issue(instance_id=instance_id,instance_data=bug, logger=logger)
#             print(explore_tree_structure(start_entities="/", traversal_depth=4, direction="both"))
#             reset_current_issue()

if __name__ == '__main__':
    """
    测试向量检索功能
    针对 ultralytics__ultralytics-17810 实例进行调试
    """
    import json
    import logging

    # ========== 配置日志 ==========
    logging.basicConfig(
        level=logging.DEBUG,  # 使用 DEBUG 级别获取更多信息
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    # ========== 测试配置 ==========
    # 测试print(semantic_search_code_snippets(**{'query': 'Initialize imgsz parameter in predictor class constructor setup method', 'file_path_or_pattern': 'ultralytics/engine/predictor.py'}))
    INSTANCE_ID = "scikit-learn__scikit-learn-28678"
    INPUT_FILE = r"D:\projects_eng\loc_proj\loc_agent\LocAgent\merged_localization_dataset.jsonl"  # 数据集文件路径
    QUERY = "function that parses and validates docstrings for parameters, attributes, and returns, stores type and description in a dictionary, and checks for identical docstrings excluding whitespaces"
    entity_name =['django/db/models/functions/datetime.py']

    # ========== 从数据集中加载实例数据 ==========
    logger.info(f"从 {INPUT_FILE} 加载实例 {INSTANCE_ID}...")

    instance_data = None
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('instance_id') == INSTANCE_ID:
                        instance_data = data
                        break
    except FileNotFoundError:
        logger.error(f"找不到文件: {INPUT_FILE}")
        logger.info("尝试手动构造实例数据...")
        # 如果找不到文件，手动构造最小必要数据
        instance_data = {
            'instance_id': INSTANCE_ID,
            'repo': 'ultralytics/ultralytics',
            'base_commit': '',  # 需要填写实际的 commit
            'problem_statement': 'Test problem statement',
        }

    if not instance_data:
        logger.error(f"在数据集中找不到实例: {INSTANCE_ID}")
        exit(1)

    logger.info(f"找到实例: {instance_data.get('instance_id')}")
    logger.info(f"Repo: {instance_data.get('repo')}")
    logger.info(f"Base commit: {instance_data.get('base_commit', 'N/A')[:8] if instance_data.get('base_commit') else 'N/A'}")

    # ========== 初始化环境 ==========
    logger.info("初始化定位环境...")
    try:
        set_current_issue(instance_data=instance_data, rank=0, logger=logger)
        logger.info("环境初始化成功")
    except Exception as e:
        logger.error(f"环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # ========== 检查全局变量状态 ==========
    logger.info("=" * 60)
    logger.info("检查全局变量状态:")
    logger.info(f"  CURRENT_ISSUE_ID: {CURRENT_ISSUE_ID}")
    logger.info(f"  CURRENT_INSTANCE: {CURRENT_INSTANCE is not None}")
    logger.info(f"  ALL_FILE: {len(ALL_FILE) if ALL_FILE else 0} 个文件")
    logger.info(f"  ALL_CLASS: {len(ALL_CLASS) if ALL_CLASS else 0} 个类")
    logger.info(f"  ALL_FUNC: {len(ALL_FUNC) if ALL_FUNC else 0} 个函数")
    logger.info(f"  DP_GRAPH: {DP_GRAPH is not None}")
    logger.info(f"  DP_GRAPH_ENTITY_SEARCHER: {DP_GRAPH_ENTITY_SEARCHER is not None}")
    logger.info("=" * 60)

    # ========== 检查向量索引 ==========
    logger.info("检查向量索引目录...")
    index_dir = os.path.join(DENSE_INDEX_DIR, INSTANCE_ID)
    logger.info(f"  索引目录: {index_dir}")
    logger.info(f"  目录存在: {os.path.exists(index_dir)}")

    if os.path.exists(index_dir):
        files_in_index = os.listdir(index_dir)
        logger.info(f"  索引目录内容: {files_in_index}")

    # ========== 测试底层向量检索 ==========
    logger.info("=" * 60)
    logger.info("测试底层向量检索 (vector_content_retrieve)...")

    # 测试get_entity_contents
    rtn = get_entity_contents(entity_name)
    print(rtn)

    try:
        # 获取文件列表
        files, _, _ = get_current_repo_modules()
        all_file_paths = [file['name'] for file in files] if files else []
        logger.info(f"  可用文件数: {len(all_file_paths)}")
        if all_file_paths:
            logger.info(f"  前5个文件: {all_file_paths[:5]}")

        # 创建 QueryInfo
        query_info = QueryInfo(term=QUERY)
        logger.info(f"  查询: {QUERY[:50]}...")

        # 直接调用 vector_content_retrieve
        logger.info("  调用 vector_content_retrieve...")
        query_results = vector_content_retrieve(query_info=query_info, include_files=all_file_paths, top_k=10)

        logger.info(f"  返回结果数: {len(query_results) if query_results else 0}")

        if query_results:
            for i, qr in enumerate(query_results):
                logger.info(f"  [{i}] nid={qr.nid}, format_mode={qr.format_mode}, "
                            f"start_line={qr.start_line}, end_line={qr.end_line}")
        else:
            logger.warning("  vector_content_retrieve 返回空结果!")

    except Exception as e:
        logger.error(f"  vector_content_retrieve 调用失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 测试更底层的 _vector_search ==========
    logger.info("=" * 60)
    logger.info("测试更底层的 _vector_search...")

    try:
        # 尝试直接加载和使用 code_index
        repo_path = None
        if CURRENT_INSTANCE and 'repo' in CURRENT_INSTANCE:
            repo_name = CURRENT_INSTANCE['repo'].replace('/', '__')
            bb_repo = r"D:\projects_eng\loc_proj\loc_agent\LocAgent\playground\repo"
            shared_repo_path = os.path.join(bb_repo, repo_name)
            if os.path.exists(shared_repo_path):
                repo_path = shared_repo_path

        if not repo_path:
            repo_path = os.path.join(bb_repo, "repo", INSTANCE_ID)

        logger.info(f"  仓库路径: {repo_path}")
        logger.info(f"  仓库存在: {os.path.exists(repo_path)}")

        # 加载向量索引
        code_index = load_vector_index(index_dir, repo_path)

        if code_index:
            logger.info("  code_index 加载成功")

            # 直接调用 _vector_search
            results = code_index._vector_search(
                query=QUERY,
                exact_query_match=False,
                category='implementation',
                file_pattern=None,
                similarity_top_k=10,
                similarity_threshold=0.4,
            )

            logger.info(f"  _vector_search 返回 {len(results)} 条结果")

            # 详细输出每条结果
            searcher = get_graph_entity_searcher()
            for i, snippet in enumerate(results):
                node = code_index._docstore.get_document(snippet.id)
                metadata = node.metadata
                file_path = metadata.get('file_path', '')
                span_ids = metadata.get('span_ids', [])
                start_line = metadata.get('start_line', 0)
                end_line = metadata.get('end_line', 0)

                logger.info(f"  [{i}] file={file_path}, span_ids={span_ids}, "
                            f"lines={start_line}-{end_line}, distance={snippet.distance:.4f}")

                # 检查 span_ids 的类型
                if span_ids:
                    all_trivial = all([sid in ['docstring', 'imports', 'comments'] for sid in span_ids])
                    any_trivial = any([sid in ['docstring', 'imports', 'comments'] for sid in span_ids])
                    logger.info(f"       all_trivial={all_trivial}, any_trivial={any_trivial}")

                    # 检查图中是否有对应节点
                    for span_id in span_ids:
                        nid = f'{file_path}:{span_id}'
                        has_node = searcher.has_node(nid) if searcher else False
                        logger.info(f"       span_id={span_id}, nid={nid}, has_node={has_node}")
        else:
            logger.error("  code_index 加载失败!")

    except Exception as e:
        logger.error(f"  _vector_search 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 测试完整的 semantic_search_code_snippets ==========
    logger.info("=" * 60)
    logger.info("测试 semantic_search_code_snippets...")

    try:
        result = semantic_search_code_snippets(query=QUERY, file_path_or_pattern="django/db/models/sql/query.py")
        logger.info(f"返回结果长度: {len(result)}")
        logger.info("=" * 40)
        logger.info("完整返回结果:")
        logger.info("=" * 40)
        print(result)
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"semantic_search_code_snippets 调用失败: {e}")
        import traceback
        traceback.print_exc()


    logger.info("=" * 60)
    logger.info("测试 search_code_snippets (关键词检索/Sparse)...")

    keyword_test_params = {
        'search_terms': ['offset_string'],
        'file_path_or_pattern': 'lib/matplotlib/dates.py',
        'retrieval_method': 'sparse' # 显式指定 sparse，对应关键词检索
    }

    try:
        logger.info(f"测试参数: {keyword_test_params}")

        # 调用 search_code_snippets
        # 注意: 原文中请求的是 keyword_search_code_snippets，但在本文件中该功能由 search_code_snippets 实现
        kw_result = search_code_snippets(**keyword_test_params)

        logger.info(f"返回结果长度: {len(kw_result)}")
        logger.info("=" * 40)
        logger.info("完整返回结果 (Keyword Search):")
        logger.info("=" * 40)
        print(kw_result)
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"search_code_snippets (keyword) 调用失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 清理环境 ==========
    logger.info("=" * 60)
    logger.info("清理环境...")
    try:
        reset_current_issue()
        logger.info("环境清理完成")
    except Exception as e:
        logger.warning(f"环境清理时出错: {e}")

    logger.info("测试完成!")
