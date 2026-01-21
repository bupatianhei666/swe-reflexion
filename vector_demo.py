import argparse
import os
import logging
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

from datasets import load_dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_CLASS, NODE_TYPE_FUNCTION
from plugins.location_tools.repo_ops.repo_ops import get_graph_entity_searcher, merge_query_results, \
    rank_and_aggr_query_results, get_repo_save_dir
from plugins.location_tools.retriever.bm25_retriever import build_retriever_from_persist_dir
from plugins.location_tools.utils.result_format import QueryResult, QueryInfo
from plugins.location_tools.utils.util import find_matching_files_from_list
from repo_index.repository import FileRepository
from repo_index.index.code_index import CodeIndex
from repo_index.index.settings import IndexSettings
from repo_index.index.simple_faiss import SimpleFaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
import dotenv
import torch

from repo_index.utils.repo import setup_repo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

dotenv.load_dotenv()

GRAPH_INDEX_DIR=os.getenv("GRAPH_INDEX_DIR")


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置日志存储路径
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(filename='vector_demo.log', encoding='utf-8'))

# def get_embed_model():
#     """获取嵌入模型"""
#     return DashScopeEmbedding(
#         model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
#         text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
#         embed_batch_size=10
#     )

def get_embed_model():
    """获取嵌入模型"""
    logger.info("Initializing HuggingFaceEmbedding model: nomic-ai/CodeRankEmbed")
    # For Jina embeddings, it's recommended to trust remote code if prompted,
    # though HuggingFaceEmbedding might handle this.
    # device="cuda" can be added if a GPU is available and preferred.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model_path = os.path.join("model", "CodeRankEmbed")
    return HuggingFaceEmbedding(
        model_name=model_path,
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
        # settings = IndexSettings(
        #     embed_model="text-embedding-v3",
        #     dimensions=1024
        # )
        settings = IndexSettings(
            embed_model="nomic-ai/CodeRankEmbed", # Descriptive name
            dimensions=768 # Dimension for jina-embeddings-v2-small-en
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


def semantic_search_code_snippets(
        code_index,
        query: str,
        instance_id = None,
        # file_path_or_pattern: Optional[str] = "**/*.py",
):
    """
    基于query进行相似度检索，在仓库中找到相关的代码块
    Args:
        query: 问题查询
        file_path_or_pattern: 筛选路径

    Returns:
    """
    # TODO: 基于问题描述的稠密检索工具
    # files, _, _ = get_current_repo_modules()
    # all_file_paths = [file['name'] for file in files]
    #
    result = ""
    # # exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    # if file_path_or_pattern:
    #     include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
    #     if not include_files:
    #         include_files = all_file_paths
    #         result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    # else:
    #     include_files = all_file_paths

    query_info = QueryInfo(term=query)
    query_results = search_relevant_code(code_index, query=query_info.term, include_files=None, instance_id = instance_id)

    result += f'##Searching for `{query}` (Semantic  Search)...\n'
    result += f"### Search Result:\n"

    query_info_list_for_ranking = [query_info]
    merged_results = merge_query_results(query_results)
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list_for_ranking)

    graph_index_file = f"{GRAPH_INDEX_DIR}/{instance_id}.pkl"
    G = pickle.load(open(graph_index_file, "rb"))
    searcher = RepoEntitySearcher(G)
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

        result += current_term_output_segment

    if not has_any_formatted_output:
        result += "No locations found after formatting.\n"

    result += '\n\n'

    if not result.strip():
        result += 'No locations found using semantic search.\n\n'
    return result.strip()


def load_bm25_retriever(bm25_index_path, repo_path, similarity_top_k = 10):
    # 加载bm25检索器
    if os.path.exists(f'{bm25_index_path}/corpus.jsonl'):
        return build_retriever_from_persist_dir(bm25_index_path)
    else:
        return None


def bm25_content_retrieve(
        instance_id,
        retriever,
        query,
        # query: str,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        similarity_top_k: int = 20
) -> str:
    """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.

    Args:
        query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
        class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
        function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
        similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.

    Returns:
        str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """
    # similarity: {score}
    cur_query_results = []
    retrieved_nodes = retriever.retrieve(query)
    graph_index_file = f"{GRAPH_INDEX_DIR}/{instance_id}.pkl"
    G = pickle.load(open(graph_index_file, "rb"))
    searcher = RepoEntitySearcher(G)

    for node in retrieved_nodes:
        file = node.metadata['file_path']
        # print(node.metadata)
        if not include_files or file in include_files:
            # drop the import code
            # if len(node.metadata['span_ids']) == 1 and node.metadata['span_ids'][0] == 'imports':
            #     continue
            if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                # TODO: drop ?
                query_result = QueryResult(query_info=query,
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
                    # searcher = get_graph_entity_searcher()
                    if searcher.has_node(nid):
                        nids.append(nid)
                    # TODO: warning if not find

                node_datas = searcher.get_node_data(nids, return_code_content=True)
                sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]

                message = ''
                if sorted_nids:
                    if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
                        nid = sorted_ndatas[0]['node_id']
                        ntype = sorted_ndatas[0]['type']
                        # The code for {ntype} {nid} is incomplete; search {nid} for the full content if needed.
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
                        nid = sorted_ndatas[-1]['node_id']
                        ntype = sorted_ndatas[-1]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if message.strip():
                        message = "Hint: \n"+ message

                nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                desc = f"Found {nids_str}."
                query_result = QueryResult(query_info=query,
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
                    # searcher = get_graph_entity_searcher()
                    if searcher.has_node(nid):
                        ndata = searcher.get_node_data([nid], return_code_content=True)[0]
                        query_result = QueryResult(query_info=query, format_mode='preview',
                                                   nid=ndata['node_id'],
                                                   ntype=ndata['type'],
                                                   start_line=ndata['start_line'],
                                                   end_line=ndata['end_line'],
                                                   retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                                   )
                        cur_query_results.append(query_result)
                    else:
                        continue

        cur_query_results = cur_query_results[:50]

        cur_result = ''
        for qr in cur_query_results:
            cur_result += qr.format_output(searcher)
            cur_result += '\n'
        print(cur_result)

        result = f'##Searching for `{query}` (Semantic  Search)...\n'
        result += f"### Search Result:\n"
        query_info = QueryInfo(term=query)
        query_info_list_for_ranking = [query_info]

        merged_results = merge_query_results(cur_query_results)
        ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list_for_ranking)

        graph_index_file = f"{GRAPH_INDEX_DIR}/{instance_id}.pkl"
        G = pickle.load(open(graph_index_file, "rb"))
        searcher = RepoEntitySearcher(G)
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

        result += current_term_output_segment

        if not has_any_formatted_output:
            result += "No locations found after formatting.\n"

        result += '\n\n'

        if not result.strip():
            result += 'No locations found using semantic search.\n\n'
        return result.strip()


def search_relevant_code(code_index, query, top_k=10, include_files=None, instance_id: str = None):
    """搜索相关代码"""
    try:
        # 使用向量搜索
        results = code_index._vector_search(
            query=query,
            exact_query_match=False,
            category='implementation',  # 只搜索实现代码，不包括测试代码
            similarity_top_k=top_k,
            similarity_threshold=0.7,
        )

        # 提取结果并转换为QueryResult对象
        cur_query_results = []
        graph_index_file = f"{GRAPH_INDEX_DIR}/{instance_id}.pkl"
        G = pickle.load(open(graph_index_file, "rb"))
        searcher = RepoEntitySearcher(G)


        for snippet in results:
            # 从 docstore 获取完整的节点信息
            node = code_index._docstore.get_document(snippet.id)
            metadata = node.metadata

            # 获取文件路径
            file = metadata.get('file_path', '')
            print(f"{file} --- distance: {snippet.distance}")

            # 检查是否在包含文件列表中
            if not include_files or file in include_files:
                # 检查是否为导入代码或文档字符串
                if 'span_ids' in metadata:
                    # 如果所有span_id都是docstring、imports或comments，作为代码片段处理
                    if all([span_id in ['docstring', 'imports', 'comments'] for span_id in metadata['span_ids']]):
                        # query_result = QueryResult(
                        #     query_info=query,
                        #     format_mode='code_snippet',
                        #     nid=file,
                        #     file_path=file,
                        #     start_line=metadata.get('start_line', 0),
                        #     end_line=metadata.get('end_line', 0),
                        #     retrieve_src=f"Retrieved code content using vector search."
                        # )
                        # cur_query_results.append(query_result)
                        continue

                    # 如果部分span_id是docstring、imports或comments
                    elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in metadata['span_ids']]):
                        nids = []
                        for span_id in metadata['span_ids']:
                            nid = f'{file}:{span_id}'
                            if searcher.has_node(nid):
                                nids.append(nid)

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
                            query_info=query,
                            format_mode='code_snippet',
                            nid=file,
                            file_path=file,
                            start_line=metadata.get('start_line', 0),
                            end_line=metadata.get('end_line', 0),
                            desc=desc,
                            message=message,
                            retrieve_src=f"Retrieved code content using vector search."
                        )

                        cur_query_results.append(query_result)
                    else:
                        # 处理其他类型的span_id
                        for span_id in metadata['span_ids']:
                            nid = f'{file}:{span_id}'
                            if searcher.has_node(nid):
                                ndata = searcher.get_node_data([nid], return_code_content=True)[0]
                                query_result = QueryResult(
                                    query_info=query,
                                    format_mode='preview',
                                    nid=ndata['node_id'],
                                    ntype=ndata['type'],
                                    start_line=ndata['start_line'],
                                    end_line=ndata['end_line'],
                                    retrieve_src=f"Retrieved code content using vector search."
                                )
                                cur_query_results.append(query_result)
                else:
                    # 处理没有span_ids的情况
                    query_result = QueryResult(
                        query_info=query,
                        format_mode='code_snippet',
                        nid=file,
                        file_path=file,
                        start_line=metadata.get('start_line', 0),
                        end_line=metadata.get('end_line', 0),
                        retrieve_src=f"Retrieved code content using vector search."
                    )
                    cur_query_results.append(query_result)

        # 限制返回结果数量
        cur_query_results = cur_query_results[:50]

        cur_result = ''
        for qr in cur_query_results:
            cur_result += qr.format_output(searcher)
            cur_result += '\n'
        print(cur_result)
        return cur_query_results

    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='测试向量索引效果')
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite",
                        help="数据集名称")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集分割")
    parser.add_argument('--index_dir', type=str, default='index_data',
                        help='向量索引的基础目录')
    parser.add_argument('--instance_id_path', type=str, default='',
                        help='包含要测试的实例ID列表的文件路径')
    parser.add_argument('--top_k', type=int, default=50,
                        help='返回的相关代码数量')
    parser.add_argument('--num_data', type=int, default=274, help='加载的数据集中的数量')
    parser.add_argument('--output_file', type=str, default='results/vector_search_results_code_rank.json',
                        help='结果输出文件')
    parser.add_argument('--repo_path', type=str, default='playground/repo', help='仓库存放路径')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 加载数据集
    dataset_name = args.dataset.split('/')[-1]
    dataset_obj = load_dataset(args.dataset, split=args.split)
    # 只加载指定的数量数据集
    dataset_obj = dataset_obj.select(range(args.num_data))

    
    # 获取实例ID列表
    if args.instance_id_path and os.path.exists(args.instance_id_path):
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
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个实例
    for instance_id in filtered_instance_ids:
        logger.info(f"处理实例: {instance_id}")
        if instance_id != 'django__django-11905':
            continue
        
        # 获取问题描述
        problem_statement = instance_data[instance_id].get('problem_statement', '')
        problem_statement = "__isnull"
        print(problem_statement)
        if not problem_statement:
            logger.warning(f"实例 {instance_id} 没有问题描述，跳过")
            continue
        
        # 构建索引路径
        index_path = os.path.join(args.index_dir, dataset_name, 'vector_index', 'coderankembed', instance_id)
        # bm25索引路径
        bm25_index_path = os.path.join(args.index_dir, dataset_name, 'BM25_index', instance_id)
        repo_path = os.path.join(args.repo_path, instance_id)

        if not os.path.exists(index_path):
            logger.warning(f"实例 {instance_id} 的索引不存在: {index_path}")
            continue
        
        # 加载向量索引
        code_index = load_vector_index(index_path, repo_path)
        # 加载bm25检索器
        retriever = load_bm25_retriever(bm25_index_path, repo_path)
        if retriever is None or not retriever:
            logger.warning(f"实例 {instance_id} 的bm25检索器不存在: {bm25_index_path}")
            continue
        if not code_index:
            logger.warning(f"实例 {instance_id} 的向量索引不存在: {index_path}")
            continue
        
        # 搜索相关代码
        relevant_code = semantic_search_code_snippets(code_index, problem_statement,  instance_id=instance_id)
        # bm25检索相关代码
        bm_relevant_code = bm25_content_retrieve(instance_id, retriever, problem_statement)

        # 存储结果
        all_results[instance_id] = {
            'problem_statement': problem_statement,
            'relevant_code': relevant_code
        }
        
        # 打印结果
        logger.info(f"实例 {instance_id} 的相关代码:")
        logger.info(relevant_code)

        logger.info(f"实例 {instance_id} 的bm25检索结果:")
        logger.info(bm_relevant_code)

        # 将 QueryResult 对象转换为可序列化的字典
        serializable_results = []
        print(relevant_code)
            # result_dict = {
            #     'query_info': result.query_info_list,
            #     'format_mode': result.format_mode,
            #     'nid': result.nid,
            #     'ntype': result.ntype if hasattr(result, 'ntype') else None,
            #     'file_path': result.file_path if hasattr(result, 'file_path') else None,
            #     'start_line': result.start_line,
            #     'end_line': result.end_line,
            #     'retrieve_src': result.retrieve_src
            # }
            # serializable_results.append(result_dict)

        # 存储结果
        # all_results[instance_id] = serializable_results

    # 保存结果到文件
    # with open(args.output_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)
    #
    # logger.info(f"结果已保存到 {args.output_file}")

if __name__ == '__main__':
    main()