import re
import json
import logging
import collections
from collections import Counter
from util.benchmark.parse_python_file import (
    parse_global_var_from_code, is_global_var
)
from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import (
    NODE_TYPE_FILE, NODE_TYPE_FUNCTION, NODE_TYPE_CLASS
)
import pickle
import os
import dotenv

dotenv.load_dotenv()

GRAPH_INDEX_DIR = os.environ.get("GRAPH_INDEX_DIR", "index_data/graph_index")


def _normalize_path_for_graph(path: str, searcher) -> str:
    """
    将路径标准化为图中实际存在的格式。
    处理 Windows 和 Linux 路径分隔符的差异。
    """
    if not path or not searcher:
        return path

    # 如果直接存在，返回原始路径
    if searcher.has_node(path):
        return path

    # 尝试将 \ 替换为 /
    normalized = path.replace('\\', '/')
    if normalized != path and searcher.has_node(normalized):
        return normalized

    # 尝试将 / 替换为 \
    normalized = path.replace('/', '\\')
    if normalized != path and searcher.has_node(normalized):
        return normalized

    return path


def _is_file_in_list(file_path: str, file_list: list) -> tuple:
    """
    检查文件路径是否在列表中（支持跨平台匹配）。
    返回 (是否匹配, 匹配到的实际路径)
    """
    if not file_path or not file_list:
        return False, None

    if file_path in file_list:
        return True, file_path

    # 标准化后比较
    normalized = file_path.replace('\\', '/')
    for f in file_list:
        if f.replace('\\', '/') == normalized:
            return True, f

    return False, None


def _path_starts_with(path: str, prefix: str) -> bool:
    """
    检查路径是否以指定前缀开始（支持跨平台匹配）。
    """
    if not path or not prefix:
        return False

    norm_path = path.replace('\\', '/')
    norm_prefix = prefix.replace('\\', '/')

    return norm_path.startswith(norm_prefix)

def parse_raw_loc_output(raw_output, valid_files):
    try:
        valid_top_folder = []
        for fn in valid_files:
            # folder = fn.split('/')[0]
            # 统一使用 / 分割
            folder = fn.replace('\\', '/').split('/')[0]
            if folder not in valid_top_folder:
                valid_top_folder.append(folder)

        # Remove the triple backticks and any surrounding whitespace
        raw_output = raw_output.strip('` \n')
        file_list, loc_edit_list = [], []

        current_file = None
        # Split the input data into lines
        lines = raw_output.strip().split('\n')
        for line in lines:
            line = line.strip().strip(':').strip()
            if not line:
                continue

            if line.endswith('.py'):
                fn = extract_python_file_path(line, valid_top_folder)
                if not fn:
                    current_file = None
                    continue

                # ✅ 修复：使用跨平台匹配
                matched, actual_path = _is_file_in_list(fn, valid_files)
                if not matched:
                    current_file = None
                    continue

                current_file = actual_path  # 使用图中的实际路径格式
                if current_file not in file_list:
                    file_list.append(current_file)

                # current_file = fn
                # if current_file not in file_list:
                #     file_list.append(current_file)

            elif current_file and line and any(
                    line.startswith(w)
                    for w in ["function:", "class:", 'method:',
                              "variable:", 'variables:', "line:", "lines:"]
            ):
                loc = f'{current_file}:{line.strip()}'
                if loc not in loc_edit_list:
                    loc_edit_list.append(loc)
                # if current_file and line not in loc_edit_dict[current_file]:
                #     loc_edit_dict[current_file].append(line)
            # 新增异常解析逻辑
            elif '.py' in line:
                # 处理django/forms/models.py:ModelMultipleChoiceField.clean这种异常格式
                parts = line.split(':')
                file_path = parts[0].strip()
                # entity_name = parts[1].strip()

                entity_name = parts[1].strip() if len(parts) > 1 else ''
                if not entity_name:  # 如果没有实体名，跳过
                    continue

                # 确保文件路径以.py结尾
                if file_path.endswith('.py'):
                    fn = extract_python_file_path(file_path, valid_top_folder)
                    if not fn:
                        continue

                    # ✅ 修复：使用跨平台匹配
                    matched, actual_path = _is_file_in_list(fn, valid_files)
                    if not matched:
                        continue

                    current_file = actual_path
                    if current_file not in file_list:
                        file_list.append(current_file)

                    # 解析实体中的类名和函数名
                    if '.' in entity_name:
                        parts = entity_name.split('.')
                        class_name = parts[0]
                        function_name = '.'.join(parts[1:])
                        class_loc = f'{current_file}:class:{class_name}'
                        if class_loc not in loc_edit_list:
                            loc_edit_list.append(class_loc)
                        function_loc = f'{current_file}:function:{function_name}'
                        if function_loc not in loc_edit_list:
                            loc_edit_list.append(function_loc)
                    else:
                        function_loc = f'{current_file}:function:{entity_name}'
                        if function_loc not in loc_edit_list:
                            loc_edit_list.append(function_loc)
        return file_list, loc_edit_list
    except Exception as e:
        logging.info("出现错误:", e)
        return [], []


def get_loc_results_from_raw_outputs(instance_id, raw_outputs, include_variable=False):
    G = pickle.load(
            open(f"{GRAPH_INDEX_DIR}/{instance_id}.pkl", "rb")
        )
    searcher = RepoEntitySearcher(G)
    all_files = searcher.get_all_nodes_by_type(NODE_TYPE_FILE)
    valid_files = [file['name'] for file in all_files]
    
    all_found_files = [[] for _ in range(len(raw_outputs))]
    all_found_modules = [[] for _ in range(len(raw_outputs))]
    all_found_entities = [[] for _ in range(len(raw_outputs))]
    for i, sample in enumerate(raw_outputs):
        found_files, found_edit_locs = parse_raw_loc_output(sample, valid_files)
        all_found_files[i] = found_files
        edit_entities = get_edit_entities_from_raw_locs(found_edit_locs, searcher,
                                                        include_variable=include_variable)
        
        filtered_edit_entities = []
        edit_modules = []
        for entity in edit_entities:
            # if entity.endswith('.__init__'):
            #     entity = entity[:(len(entity)-len('.__init__'))]
            # 恢复注释
            raw_entity = entity
            if entity.endswith('.__init__'):
                entity = entity[:(len(entity)-len('.__init__'))]

            # ✅ 修复：标准化路径后检查
            normalized_entity = _normalize_path_for_graph(entity, searcher)

            if searcher.has_node(normalized_entity):
                if raw_entity.endswith('.__init__'):
                    # 这里将__init__方法定位结果加入答案
                    filtered_edit_entities.append(raw_entity)

                entity_data = searcher.get_node_data([normalized_entity])[0]
                if entity_data['type'] == NODE_TYPE_FUNCTION:
                    filtered_edit_entities.append(normalized_entity)
                # 这里最起码要把模块定位结果加入
                elif entity_data['type'] == NODE_TYPE_CLASS:
                    edit_modules.append(normalized_entity)
            else:
                continue

            if '.' in normalized_entity.split(':')[-1]:
                parent_entity = '.'.join(normalized_entity.split('.')[:-1])
                normalized_parent = _normalize_path_for_graph(parent_entity, searcher)
                if searcher.has_node(normalized_parent):
                    if normalized_parent not in edit_modules:
                        edit_modules.append(normalized_parent)
            else:
                if normalized_entity not in edit_modules:
                    edit_modules.append(normalized_entity)
            
        all_found_entities[i] = filtered_edit_entities
        all_found_modules[i] = edit_modules
    return all_found_files, all_found_modules, all_found_entities


def extract_python_file_path(line, valid_folders):
    """
    Extracts the Python file path from a given line of text.

    Parameters:
    - line (str): A line of text that may contain a Python file path.

    Returns:
    - str or None: The extracted Python file path if found; otherwise, None.
    """
    # Define a regular expression pattern to match file paths ending with .py
    # The pattern looks for sequences of characters that can include letters, numbers,
    # underscores, hyphens, dots, or slashes, ending with '.py'
    pattern = r'[\w\./-]+\.py'

    # Search for the pattern in the line
    match = re.search(pattern, line)

    if match:
        matched_fp = match.group(0)
        start_index = len(matched_fp)
        for folder in valid_folders:
            if f'{folder}/' in matched_fp:
                cur_start_index = matched_fp.index(f'{folder}/')
                if cur_start_index < start_index:
                    start_index = cur_start_index
        if start_index < len(matched_fp):
            return matched_fp[start_index:] # Return the max matched file path
        return None
    else:
        return None  # Return None if no match is found


def merge_sample_locations(found_files, found_modules, found_entities, ranking_method='majority'):
    
    def rank_locs(found_locs, ranking_method="majority"):
        flat_locs = [loc for sublist in found_locs for loc in sublist]
        # unique_files = list(set(flat_files))
        locs_weights = collections.defaultdict(float)
        # ranked_locs = list()
        
        if ranking_method == "majority":
            """Rank files based on their frequency of occurrence"""
            loc_counts = Counter(flat_locs)
            for loc, count in loc_counts.items():
                locs_weights[loc] = count
        
        elif ranking_method == "mrr":
            """Rank files based on Mean Reciprocal Rank (MRR) of their edit locations"""
            # Calculate MRR for the edit locations: sum of (1 / rank)
            for sample_locs in found_locs:
                for rank, loc in enumerate(sample_locs, start=1):
                    locs_weights[loc] += 1 / rank
        
        # Rank the files based on the selected ranking method
        ranked_loc_weights = sorted(locs_weights.items(), key=lambda x: x[1], reverse=True)
        ranked_locs = [file for file, _ in ranked_loc_weights]
        return ranked_locs, ranked_loc_weights

    # Rank files
    ranked_files, file_weights = rank_locs(found_files, ranking_method)
    ranked_modules, module_weights = rank_locs(found_modules, ranking_method)
    ranked_funcs, func_weights = rank_locs(found_entities, ranking_method)
    
    return ranked_files, ranked_modules, ranked_funcs
    

# def get_edit_modules_from_file_to_dict(pred_files, file_to_edit_locs, structure, keep_whole_class=False):
def get_edit_entities_from_raw_locs(found_edit_locs, 
                                    searcher: RepoEntitySearcher,
                                    ranking_method='mrr',
                                    include_variable=False,
                                    ):
    # topn locs
    found_edit_entities = []
    current_class_name = ""
    prev_file_name = ""
    for i, edit_loc in enumerate(found_edit_locs):
        pred_file = edit_loc.split(':')[0].strip()
        if prev_file_name and prev_file_name != pred_file:
            current_class_name = ""
        prev_file_name = pred_file
        
        loc = ':'.join(edit_loc.split(':')[1:]).strip()
        # i = pred_files.index(pred_file)

        # ✅ 修复：标准化文件路径
        normalized_pred_file = _normalize_path_for_graph(pred_file, searcher)
        
        # get file content -> parse global var
        if searcher.has_node(normalized_pred_file):
            pred_file_content = searcher.G.nodes[normalized_pred_file]['code']
            global_vars = parse_global_var_from_code(pred_file_content)
        else:
            continue
        
        if loc.startswith("line:") or loc.startswith("lines:"):
            loc = loc.split(":")[1].strip()
            pred_lines = []
            # Regular expression to match different line formats
            # match = re.match(r"\s*(\d+)\s*[-ｰ]?\s*(\d+)?", loc)
            matches = re.findall(r'\s*(\d+)(?:-(\d+))?', loc)
            for match in matches:
                start_line = max(1, int(match[0]))
                end_line = int(match[1]) if match[1] else start_line
                end_line = min(len(pred_file_content.splitlines()), end_line)
                pred_lines += list(range(start_line, end_line+1))
            if not matches:
                loc = loc.split()[0]
                try:
                    pred_lines.append(int(loc.strip()))
                except:
                    logging.debug(f'line {loc} not found')
            
            pred_lines = list(set(pred_lines))
            pred_lines.sort()
            cur_found_modules = get_modules_from_line_numbers(pred_lines, normalized_pred_file, searcher,
                                                              global_vars,
                                                              include_variable=include_variable)
            for cmodule in cur_found_modules:
                if cmodule['type'] == NODE_TYPE_CLASS:
                    # 这里有问题，cmoudule是个字典！！！
                    # current_class_name = cmodule.split(':')[-1].strip()
                    current_class_name = cmodule['name'].split(':')[-1].strip()
                    
                if cmodule['type'] == NODE_TYPE_FUNCTION:
                    found_edit_entities.append(cmodule['name'])
        
        # handle cases like "class: MyClass"
        elif loc.startswith("class:") and "." not in loc:
            loc = loc[len("class:") :].strip()
            loc = loc.split()[0]
            module_id = f'{normalized_pred_file}:{loc.strip()}'  # ✅ 使用标准化路径

            # ✅ 修复：标准化后检查
            normalized_module_id = _normalize_path_for_graph(module_id, searcher)
            if normalized_module_id in searcher.G:
                current_class_name = loc
            else:
                logging.info(f"{loc} class could not be found")

            # module_id = f'{pred_file}:{loc.strip()}'
            # if module_id in searcher.G:
            #     current_class_name = loc
            # else:
            #     logging.info(f"{loc} class could not be found")
                
        elif loc.startswith("function: ") or loc.startswith("method: ") or "." in loc:
            full_loc = loc
            loc = loc.split(":", 1)[-1].strip('() ')
            loc = loc.split()[0]


            # handle cases like "function: MyClass.my_method"/ "class: MyClass.my_method"
            # for cases like "function: MyClass.my_method.inner_method", ignore "inner_method"
            if "." in loc:
                class_name = loc.split(".")[0]
                method_name = loc.split(".")[1]

                module_id = f'{normalized_pred_file}:{class_name}.{method_name}'  # ✅
                if method_name == '__init__':
                    module_id = f'{normalized_pred_file}:{class_name}'

                # ✅ 修复：标准化后检查
                normalized_module_id = _normalize_path_for_graph(module_id, searcher)
                if normalized_module_id in searcher.G:
                    found_edit_entities.append(normalized_module_id)
                    continue
                else:
                    logging.debug(f"{full_loc} method could not be found")


            # ✅ 修复：标准化后检查
            direct_module_id = f"{normalized_pred_file}:{loc}"
            normalized_direct = _normalize_path_for_graph(direct_module_id, searcher)

            # directly search for the function 'loc'
            if normalized_direct  in searcher.G:
                found_edit_entities.append(normalized_direct)
            # relevant_function = get_function_by_name(loc, pred_file, functions=functions)
            else:
                logging.debug(f"{loc} function could not be found")

                if current_class_name != "":
                    class_module_id = f"{normalized_pred_file}:{current_class_name}"
                    normalized_class = _normalize_path_for_graph(class_module_id, searcher)

                    if normalized_class in searcher.G:
                        potential_class = searcher.get_node_data([normalized_class])[0]
                        if potential_class['type'] == NODE_TYPE_CLASS:
                            _loc = loc
                            if '.' in loc:
                                _loc = _loc.split('.')[-1]
                            _module_id = f"{normalized_pred_file}:{current_class_name}.{_loc}"
                            normalized_method = _normalize_path_for_graph(_module_id, searcher)

                            if normalized_method not in searcher.G:
                                logging.debug(f"{_module_id}找不到，直接将其作为结果")
                            found_edit_entities.append(normalized_method)
                else:
                    if loc in searcher.global_name_dict:
                        nids = searcher.global_name_dict[loc]
                        cadidate_nids = []
                        for nid in nids:
                            # ✅ 修复：使用跨平台路径匹配
                            if _path_starts_with(nid, normalized_pred_file):
                                cadidate_nids.append(nid)
                        if len(cadidate_nids) == 1:
                            found_edit_entities.append(cadidate_nids[0])
            # else:
            #     found_edit_modules.append(f'{pred_file}:{loc}')
        # - end identify function -

        elif include_variable and loc.startswith(("variable:", "variables:")):
            vars = loc.split(':')[-1].strip().replace(',', ' ').split()
            for v in vars:
                if global_vars and v in global_vars:
                    found_edit_entities.append(f'{normalized_pred_file}:{v}')
        else:
            if loc.strip():
                logging.info(f"loc {loc} not recognised")
    
    
    loc_weights = collections.defaultdict(float)
    # Apply the selected merging method
    if ranking_method == "majority":
        # Majority Voting: Count the frequency of each edit location
        loc_counts = Counter(found_edit_entities)
        for loc, count in loc_counts.items():
            loc_weights[loc] = count
    elif ranking_method == "mrr":
        for rank, loc in enumerate(found_edit_entities, start=1):
            # Calculate MRR for edit locations
            loc_weights[loc] += 1 / rank
            
    # Sort edit locations based on weight
    ranked_loc_weights = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)
    res_edit_entities = [loc for loc, _ in ranked_loc_weights]
    # found_edit_module_loc = [['\n'.join(modules)] for modules in found_edit_modules]
    # import pdb; pdb.set_trace()
    return res_edit_entities


def get_modules_from_line_numbers(line_numbers, 
                                  pred_file, 
                                  searcher: RepoEntitySearcher,
                                  global_vars: dict=None,
                                  include_variable: bool = False,
                                  ):
    found_mnames, found_mnodes = [], []
    cur_module_end_line = None
    for line in line_numbers:
        # TODO: check if global var
        # if include_variable and global_vars:
        #     variable = is_global_var(line, global_vars)
        #     if variable and variable not in found_modules:
        #         # found_modules.append(f"variable: {variable}")
        #         found_modules.append(f"{pred_file}:{variable}")
        #         continue
        if cur_module_end_line and line <= cur_module_end_line:
            continue
        module, cur_module_end_line = get_module_from_line_number(line, pred_file, searcher)
        if module and module['name'] not in found_mnames:
            found_mnames.append(module['name'])
            found_mnodes.append(module)
    return found_mnodes


def get_module_from_line_number(line, file_path, searcher):
    # ✅ 修复：标准化路径后检查
    normalized_file_path = _normalize_path_for_graph(file_path, searcher)

    if normalized_file_path not in searcher.G.nodes:
        logging.warning(f"File path not found in graph: {file_path}")
        return (None, None)

    file_node = searcher.get_node_data([normalized_file_path])[0]
    cur_start_line = file_node['start_line']
    cur_end_line = file_node['end_line']
    cur_node = None

    for nid in searcher.G.nodes():
        # ✅ 修复：使用跨平台路径匹配
        if not _path_starts_with(nid, normalized_file_path):
            continue

        node = searcher.G.nodes[nid]
        if 'start_line' in node and 'end_line' in node:
            if node['start_line'] < cur_start_line or node['end_line'] > cur_end_line:
                continue
            if line >= node['start_line'] and line <= node['end_line']:
                cur_node = dict(node)
                cur_node['name'] = nid
                cur_start_line = node['start_line']
                cur_end_line = node['end_line']

    if cur_node:
        return (cur_node, cur_end_line)
    return (None, None)


if __name__ == "__main__":
    # 测试get_loc_results_from_raw_outputs函数
    # 注意：这里输出的是单引号，需要将其转化为双引号，否则会报错
    instance_id = "django__django-13448"
    raw_outputs = ["""```
django/db/backends/base/creation.py
line: 61
line: 77-78
class: BaseDatabaseCreation
function: BaseDatabaseCreation.create_test_db

django/db/backends/base/creation.py
line: 94-122
class: BaseDatabaseCreation
function: BaseDatabaseCreation.serialize_db_to_string

django/db/backends/base/creation.py
line: 101-118
class: BaseDatabaseCreation
function: BaseDatabaseCreation.serialize_db_to_string.get_objects

django/test/utils.py
line: 174
function: setup_databases

django/test/runner.py
line: 616
class: DiscoverRunner
function: DiscoverRunner.setup_databases
```"""]
    # 将当前代码运行目录切换到项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(instance_id, raw_outputs)
    # print(f"all_found_files: {all_found_files}")
    # print(f"all_found_modules: {all_found_modules}")
    # print(f"all_found_entities: {all_found_entities}")
    print(f"all_found_files: {json.dumps(all_found_files)}")
    print(f"all_found_modules: {json.dumps(all_found_modules)}")
    print(f"all_found_entities: {json.dumps(all_found_entities)}")