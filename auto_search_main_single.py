import argparse
import os
import json
import logging
import logging.handlers
import time
import toml
from queue import Empty, Queue
from typing import List
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset

from util.prompts.pipelines.critic_prompt import CRITIC_INSTRUCTION
from util.prompts.pipelines.re_explore_prompt import ReExplorePrompt, NoCriticExplorePrompt
from util.runtime.execute_ipython import execute_ipython
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
    new_task_prompt as new_task,
    collaborative_auto_search_prompt as collab_auto_search,
)
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
)
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError
from openai import OpenAI
# import openai
import dotenv

dotenv.load_dotenv()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

client = OpenAI(
    base_url=os.environ['OPENAI_API_BASE'],
    api_key=os.environ['OPENAI_API_KEY']
)

# BASE_URL=os.getenv('MY_BASE_URL', '')
# API_KEY=os.getenv('MY_API_KEY', '')

# litellm._turn_on_debug()

from time import sleep
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function)
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset


def get_task_instruction(instance: dict, task: str = 'auto_search', include_pr=False, include_hint=False, explore_experience=None):
    output_format = None
    instruction = ""
    
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = collab_auto_search.TASK_INSTRUECTION.format(
            package_name=instance['instance_id'].split('_')[0]
        )
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC

    elif task.strip() == 'new_task':
        task_description = new_task.NEW_TASK_INSTRUECTION.format(
            explore_experience=explore_experience,
            package_name=instance['instance_id'].split('_')[0]
        )
    else:
        return None

    instruction += task_description
        
    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description = '\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )
    
    if output_format:
        instruction += output_format
    
    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    # NOTE: You can actually set slightly different instruction for different task
    # instruction += AGENT_CLS_TO_INST_SUFFIX
    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        tools = None,
                        traj_data=None,
                        temp=0,
                        max_iteration_num=60,
                        use_function_calling=True,
                        instance_id=None,
                        logger=None,
                        bug=None,
                        critic_agent=True,
                        reflection_agent=True):
    if tools and ('hosted_vllm' in model_name
                  # or 'qwen' in model_name.lower()
    #             #   or model_name=='azure/gpt-4o' 
    #             #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                ):
        use_function_calling = False
        
    # for LLM which do not support function calling
    if not use_function_calling:
        # 转换message
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
        
    cur_interation_num = 0
    tool_iteration_num = 0
    last_message = None
    # 为了防止死循环，限制最大last_message重复次数
    max_last_message_repeat_num = 0
    finish = False

    """
        exploration_history记录agent的使用工具的探索历史，供批判agent进行合理性判断
        需要记录每次工具调用的历史，将工具调用转化为字符串，只需要记录工具名称以及参数信息，
        例如:
        save_thought(message="This is thought text")
        keyword_search_code_snippets({'search_terms': ['PolynomialError', 'Piecewise generators do not make sense']})
        semantic_search_code_snippets({'query': 'polynomial system initialization with piecewise expressions'})
    """
    exploration_history = []
    tool_call_experiences = []
    """
        backtrack_messages用于记录exploration_history中工具调用记录编号到相应的上下文messages的映射
        当执行者决定需要重新指定执行计划时，不是从头开始进行缺陷定位推理而是，根据批判者的回溯反馈结果进行回溯
        批判者会返回回溯位置的对应的工具调用记录编号，从backtrack_messages取编号对应的messages即可完成上下文会话记录的回溯
        结构如下：
        {
            "1": {
                "tool": "save_thought(...)",
                "messages": [
                    {"role":...}
                ]
            }
        }
    """
    # 保存exploration_history到干净回溯历史id的映射
    backtrack_messages = {}
    # clean_backtrach_messages仅记录干净的回溯历史，不包括ReExplorePrompt这些
    clean_backtrach_messages = {
        "0": messages.copy()
    }
    clean_back_cnt = 0
    exploration_idx = 1
    back_id = None
    last_action = 0
    back_cnt = 0
    backtrack_messages[str(exploration_idx)] = str(clean_back_cnt)

    while not finish:
        cur_interation_num += 1
        # 为了避免大模型无限循环，设置最大迭代次数
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of iteration has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of iteration has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            logger.info(f"达到最大迭代次数")
        elif cur_interation_num > max_iteration_num:
            logger.info(f"达到最大迭代次数")
            break

        try:
            # new conversation
            if tools and ('hosted_vllm' in model_name
                          # or 'qwen' in model_name.lower()
            ):
                messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    # api_key=API_KEY,
                    # api_base=BASE_URL,
                    temperature=temp, top_p=0.8, repetition_penalty=1.05, 
                    messages=messages,
                    stop=NON_FNCALL_STOP_WORDS,
                    stream=True if model_name == 'openai/qwen3-235b-a22b' else False
                )
            elif tools:
                response = litellm.completion(
                    model=model_name,
                    # api_key=API_KEY,
                    # api_base=BASE_URL,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                    stream=True if model_name == 'openai/qwen3-235b-a22b' else False
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    # api_key=API_KEY,
                    # api_base=BASE_URL,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                    stream=True if model_name == 'openai/qwen3-235b-a22b' else False
                )
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            result_queue.put({'error': str(e), 'type': 'BadRequestError'})
            return
        # 记录messages的数量，判断是否有对messages进行回溯
        logger.info(f"\nmessages length: {len(messages)}\n")
        if last_message and response.choices[0].message.content == last_message and max_last_message_repeat_num < 3:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            max_last_message_repeat_num += 1
            # 如果达到两次重复，则结束对话
            if max_last_message_repeat_num == 3:
                logger.info(f"max_last_message_repeat_num: {max_last_message_repeat_num}")
                finish = True
            continue
        
        raw_response = deepcopy(response)
        # logging.info('response.choices[0].message')
        # deepseek已经支持函数调用了
        # TODO：需要修改，因为deepseek已经支持函数调用了
        if tools and ('hosted_vllm' in model_name
                      # or 'qwen' in model_name.lower()
                    #   or 'deepseek' in model_name
                      ):
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except:
                logger.info('convert none fncall messages failed.')
                continue 
                
        last_message = response.choices[0].message.content

        # 将模型的输出记录到日志中
        logger.info(f"Model Response:\n {response.choices[0].message}")
        messages.append(convert_to_json(raw_response.choices[0].message))
        system_role_message = convert_to_json(raw_response.choices[0].message)
        system_role_add_cnt = 0

        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens  
            
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        logger.info(f"actions长度：{len(actions)}")
        for action in actions:
            logger.info("action:")
            logger.info(action)
            logger.info("action.action_type:")
            logger.info(action.action_type)
            if action.action_type == ActionType.FINISH:
                # finish tool无需记录工具调用历史，因为到这就结束了
                final_output = action.thought
                logger.info('='*15)
                logger.info("\n" + instance_id + " Final Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                # 无需记录工具调用历史
                logger.info("thought:\n" + action.content)
                # check if enough
                # 判断action.content是否满足输出格式，即为```包围的字符串
                # if action.content.strip().startswith('```') and action.content.strip().endswith('```'):
                #     logger.info(f"提前结束：\n{action.content}")
                #     final_output = action.content
                    # finish = True
                # messages.append({"role": "user", "content": fake_user_msg})
                # traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
                # 解析content内容
                message_content = action.content
                if message_content.startswith('```json'):
                    message_content = message_content[7:]
                if message_content.endswith('```'):
                    message_content = message_content[:-3]
                # 尝试将对字符串以json格式解析，如果出错不是Json格式，则跳过
                try:
                    message_content = json.loads(message_content)
                    decision = message_content['decision']
                    # 避免重复输出{}定义
                    if last_action == 1:
                        # 抛出异常
                        # finish = True
                        # raise Exception('重复输出{}定义')
                        # break
                        logger.warning("Model has output a decision JSON again, despite instructions. Adding a corrective message and retrying.")
                        messages.append({
                            "role": "user",
                            "content": "OBSERVATION: You have already made a decision. You MUST now proceed with your previous decision by using a tool. Your next response should be a tool call."
                        })
                        # Reset the flag to allow recovery without getting stuck in a loop.
                        continue

                    if reflection_agent:
                        # 如果decision为CHANGE_STRATEGY，则将当前路径的经验提取出来
                        if decision == 'CHANGE_STRATEGY':
                            # 如果back_id为None则continue
                            if back_id is None:
                                continue

                            experience = message_content['experience_summary']
                            # 重新选择推理路径，需要将当前上下文清空，将当前工具探索路径的经验添加到经验集中
                            """
                                将exploration_history中的字符串拼接成以下样式：
                                toolcall1() --> toolcall2()...
                            """
                            exploration_history_str = "** TOOL_CALL_CHAINS: "
                            exploration_history_str += ' --> '.join(exploration_history)
                            exploration_history_str = exploration_history_str + "\n** EXPERIENCE: " +  experience
                            # 拼接初始提示词
                            # messages = []
                            # messages.append({
                            #     "role": "user",
                            #     "content": get_task_instruction(bug, task="new_task", include_pr=True, include_hint=True, explore_experience=exploration_history_str),
                            # })
                            # 打印新任务日志
                            # logger.info(f"新任务指令:\n {get_task_instruction(bug, task="new_task", include_pr=True, include_hint=True, explore_experience=exploration_history_str)}")
                            # 将exploration_history清空
                            # exploration_history = []

                            # 智能体合作模式，进行messages回溯，同时将经验作为其他智能体得到的经验总结
                            # messages = backtrack_messages[str(back_id)]['messages']
                            past_messages = clean_backtrach_messages[backtrack_messages[str(back_id)]]
                            messages = deepcopy(past_messages)

                            # 注入反馈信息定义新的、更精确的提示词模板
                            # 这个提示词指导智能体如何利用新信息来调整其下一步行动，而不是重置整个计划
                            feedback_prompt_template = """
--- STRATEGIC UPDATE FROM A COLLABORATING AGENT ---
                        
## Context: ##
You are at a key decision point in your exploration. A parallel exploration conducted by another agent, which started from your current state, has been identified as an ineffective path. Your history up to this point remains valid.

## Summary of the Ineffective Path: ##
-- BEGIN OF INEFFECTIVE PATH --
{ineffective_path_summary}
-- END OF INEFFECTIVE PATH --

## Your Task: ##
1.  **Internalize the Lesson:** Review the `TOOL_CALL_CHAINS` and the `EXPERIENCE` summary above to understand precisely *why* this subsequent path failed.
2.  **Adjust Your Next Step:** You must incorporate this new knowledge into your reasoning. Your next `thought` should acknowledge this feedback and articulate a different approach *moving forward from this point*. You do not need to restart your entire plan.
3.  **Continue the Mission:** Proceed with your localization task, using this lesson to guide your subsequent actions and avoid the previously identified dead end.Please use the save_thought tool to summarize and memorize the feedback from the collaborative agent first.
                            """

                            # 使用具体的失败路径信息格式化提示词
                            feedback_message_content = feedback_prompt_template.format(
                                ineffective_path_summary=exploration_history_str
                            )
                            logger.info(f"feedback_message_content:\n{feedback_message_content}")

                            # 将新的指令性消息添加到已恢复的消息历史记录中
                            # 这将成为 LocAgent 看到的下一个输入
                            messages.append({
                                "role": "user",
                                "content": feedback_message_content,
                            })
                            # cur_interation_num减小
                            cur_interation_num = cur_interation_num - len(exploration_history) + back_id
                            # 将exploration_history回溯到id对应的位置，注意id是从1开始的
                            exploration_history = exploration_history[:back_id + 1]
                            # 将历史经验保存到backtrack_messages中
                            # TODO: 历史经验保存的有问题
                            # backtrack_messages[str(back_id)]['messages'] =  messages

                            # 将历史经验加入到干净会话历史中
                            # past_meassgaes = clean_backtrach_messages[str(clean_back_cnt)]
                            clean_back_cnt = clean_back_cnt + 1
                            past_messages.append({
                                "role": "user",
                                "content": feedback_message_content,
                            })
                            clean_backtrach_messages[str(clean_back_cnt)] = deepcopy(past_messages)
                            backtrack_messages[str(back_id)] = str(clean_back_cnt)

                            back_cnt = back_cnt + 1
                            if back_cnt > 10:
                                # 避免死循环
                                messages.append({"role": "user", "content": fake_user_msg})
                                traj_msgs.append({"role": "user", "content": fake_user_msg})
                                if back_cnt > 11:
                                    finish = True
                                logger.info("back_cnt > 15, ready to stop")
                            last_action = 1
                            continue
                        elif decision == 'CONTINUE_CURRENT_PATH':
                            # 继续当前推理路径
                            messages.append({
                                "role": "user",
                                "content": message_content['reasoning']
                            })
                            last_action = 1
                    else:
                        # 消融实验，仅仅让模型根据自身当前上下文进行经验总结
                        no_critic_feedback_prompt_template = """
--- Experience Summary and Adjustment ---

## Context: ## 
You are at a key decision point. You need to summarize your experience and adjust your strategy based on the context you just received.

## Your Task: ## 
1. **Review and Reflect:** Carefully evaluate the context you *just* received (e.g., tool outputs, previous reasoning steps, or obstacles encountered). 
2. **Summarize Experience:** What does this new information tell you? What worked? What didn't? What is the key lesson learned? 
3. **Adjust Next Step:** Based on this new understanding, how should your plan for the next step be adjusted?

Please use the save_thought tool first to summarize and memorize the lesson you have derived from the current context.
                        """
                        logger.info("self summarize begin")
                        messages.append({
                            "role": "user",
                            "content": no_critic_feedback_prompt_template,
                        })
                        last_action = 1
                        continue
                except Exception as e:
                    logger.info('message_content is not json format.')
                    logger.info(e)
                    break

            # 新增对thought action的解析
            elif action.action_type == ActionType.THOUGHT:
                thought = action.thought
                # 记录save_thought工具调用
                tool_history = f'save_thought(message="{thought}")"'
                exploration_history.append(tool_history)
                logger.info("thought:\n" + thought)
                messages.append({
                    "role": "tool",
                    "tool_call_id": action.tool_call_id,
                    "content": "THOUGHT: \n" + thought,
                })
                traj_msgs.append({
                    "role": "tool",
                    "tool_call_id": action.tool_call_id,
                    "content": "THOUGHT: \n" + thought,
                })

                # 将当前对话加入到干净回溯历史中
                past_messages = clean_backtrach_messages[str(clean_back_cnt)]
                clean_back_cnt = clean_back_cnt + 1
                if system_role_add_cnt == 0:
                    past_messages.append(system_role_message)
                    system_role_add_cnt = system_role_add_cnt + 1

                past_messages.append({
                    "role": "tool",
                    "tool_call_id": action.tool_call_id,
                    "content": "THOUGHT: \n" + thought,
                })
                clean_backtrach_messages[str(clean_back_cnt)] = deepcopy(past_messages)

                # 保存回溯上下文信息
                backtrack_messages[str(exploration_idx)] = str(clean_back_cnt)
                exploration_idx = exploration_idx + 1
                last_action = 0
                # messages.append({
                #     "role": "user",
                #     "content": thought
                # })
                # traj_msgs.append({
                #     "role": "user",
                #     "content": thought
                # })
            elif action.action_type == ActionType.RUN_IPYTHON:
                # 执行ipython代码
                tool_iteration_num = tool_iteration_num + 1
                ipython_code = action.code.strip('`')
                # 这个ipython_code是print(expl...)形式，需要去除其左边的print(和右边的最后一个)
                tool_history = ipython_code.lstrip('print(')[:-1]
                # 记录工具调用
                exploration_history.append(tool_history)
                logger.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                last_action = 0
                # logger.info(f"Function response:\n{function_response}")
                try:
                    """
                    以下代码执行出错：
                    print(explore_tree_structure(**{'start_entities': ['src/_pytest/logging.py'], 'traversal_depth': 1, 'dependency_type_filter': ['imports']}))
                    报错：
                    NameError: name 'src' is not defined
                    """
                    # 正常来说，funtion_response已经是执行图遍历之后的字符串展示结果，但是如果执行图遍历之后返回的仅仅是路径，那么就会报错
                    function_response = eval(function_response)
                    logger.info(f"代码执行结果：\n{function_response}")
                except SyntaxError:
                    function_response = function_response
                except NameError:
                    # 新增NameError处理逻辑
                    function_response = function_response
                except Exception as e:
                    logger.info(f"代码执行出错：\n{e}")
                    # 结束当前任务
                    finish = True
                    final_output=None
                #TODO: NameError报错处理
                # except NameError:
                #     logging.info("NameError:\n")
                #     function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)

                logger.info("OBSERVATION:\n" + function_response)
                # exploration_history.append("TOOL RESPONSE:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })

                    # 将工具调用信息加入到干净回溯历史中
                    past_messages = clean_backtrach_messages[str(clean_back_cnt)]
                    clean_back_cnt = clean_back_cnt + 1
                    if system_role_add_cnt == 0:
                        past_messages.append(system_role_message)
                        system_role_add_cnt = system_role_add_cnt + 1
                    past_messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    clean_backtrach_messages[str(clean_back_cnt)] = deepcopy(past_messages)
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })

                    # 将工具调用信息加入到干净回溯历史中
                    past_messages = clean_backtrach_messages[str(clean_back_cnt)]
                    clean_back_cnt = clean_back_cnt + 1
                    if system_role_add_cnt == 0:
                        past_messages.append(system_role_message)
                        system_role_add_cnt = system_role_add_cnt + 1
                    past_messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    clean_backtrach_messages[str(clean_back_cnt)] = deepcopy(past_messages)

                # 保存回溯上下文信息
                backtrack_messages[str(exploration_idx)] = str(clean_back_cnt)
                exploration_idx = exploration_idx + 1

            else:
                logger.warning('Error Action!')
                # return

        # 如果cur_interation_num%5==0，需要批判agent根据当前推理路径做出评价，避免陷入无效重试循环以及局部最优解
        # 这里需要做消融实验，tool_iteration_num为1、3、5、7
        if tool_iteration_num % 9 == 0 and tool_iteration_num > 0:
            if critic_agent:
                # 将exploration_history拼接成1....\n2....的形式
                historys = ""
                idx = 0
                for i, tool_history in enumerate(exploration_history):
                    if not tool_history.startswith("TOOL RESPONSE"):
                        idx = idx + 1
                    else:
                        historys += "\t" + tool_history + "\n"
                        continue
                    historys += str(idx) + ". --" + tool_history + "\n"

                critic_prompt = CRITIC_INSTRUCTION.format(
                    problem_statement=bug['problem_statement'],
                    explortaion_history=historys,
                )
                critic_messages = [
                    {
                        "role": "system",
                        "content": "You are an expert debugger and senior architect, responsible for critiquing an AI's debugging process and recommending more effective strategies."
                    },
                    {
                        "role": "user",
                        "content": critic_prompt,
                    }
                ]
                critic_response = litellm.completion(
                    model=model_name,
                    # api_key=API_KEY,
                    # api_base=BASE_URL,
                    messages=critic_messages,
                    temperature=1,
                )
                logger.info("critic_prompt:\n" + critic_prompt)
                logger.info(f"critic_response:\n{critic_response.choices[0].message}")
                # 解析批判agent的输出结果
                critic_response_text = critic_response.choices[0].message.content.strip()
                if critic_response_text.startswith('```json'):
                    critic_response_text = critic_response_text[7:]
                if critic_response_text.endswith('```'):
                    critic_response_text = critic_response_text[:-3]
                critic_data = json.loads(critic_response_text)
                # 在日志中以易读的形式打印critic_data
                logger.info(f"critic_data:\n{json.dumps(critic_data, indent=4)}")
                # 解析出critic_data中的evaluation的reasoning和recommendation的reasoning以及next_steps字段内容，并通过\n进行拼接
                critic_str = critic_data['evaluation']['reasoning']
                effective_flag = critic_data['evaluation']['effectiveness']
                # 获取回溯id
                back_id = critic_data['evaluation']['backtrack_to_id']
                if not effective_flag == 'Effective':
                    if reflection_agent:
                        # messages.append({
                        #     "role": "user",
                        #     "content": critic_str,
                        # })
                        meta_prompt = ReExplorePrompt.format(
                            critic_str=critic_str
                        )
                        logger.info("meta_prompt:\n" + meta_prompt)
                        messages.append({"role": "user", "content": meta_prompt})
                    else:
                        # 仅仅有批判agent但是不进行经验总结，直接将上下文回溯，并将批判者的原因加入到上下文
                        past_messages = clean_backtrach_messages[backtrack_messages[str(back_id)]]
                        messages = deepcopy(past_messages)

                        no_critic_prompt = """
Use the save_thought tool to formulate a new execution plan.
                        """
                        messages.append({"role": "user", "content": critic_str+no_critic_prompt})
                        cur_interation_num = cur_interation_num - len(exploration_history) + back_id
                        exploration_history = exploration_history[:back_id + 1]
                        # clean_back_cnt = clean_back_cnt + 1
                        # past_messages.append({
                        #     "role": "user",
                        #     "content": feedback_message_content,
                        # })
                        # clean_backtrach_messages[str(clean_back_cnt)] = deepcopy(past_messages)
                        # backtrack_messages[str(back_id)] = str(clean_back_cnt)

                        back_cnt = back_cnt + 1
                        if back_cnt > 10:
                            # 避免死循环
                            messages.append({"role": "user", "content": fake_user_msg})
                            traj_msgs.append({"role": "user", "content": fake_user_msg})
                            if back_cnt > 11:
                                finish = True
                            logger.info("back_cnt > 15, ready to stop")


                # 将exploration_history清空
                # exploration_history = []
            else:
                # 消融实验，去除批判agent，仅仅包括反省机制，即让探索agent进行周期性反省并且总结当前路径探索经验
                if reflection_agent:
                    self_critic_messages = {
                            "role": "user",
                            "content": NoCriticExplorePrompt
                    }

                    logger.info("self critic begin")
                    messages.append(self_critic_messages)
            tool_iteration_num = tool_iteration_num + 1
    # save traj
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    }
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")

    while True:
        try:
            # 非阻塞获取队列中的bug
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
            agent_skills_docs=LocationToolsRequirement.documentation,
        )

        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        set_current_issue(instance_data=bug, rank=rank)

        # loc result
        raw_output_loc = []
        loc_trajs = {'trajs': []}
        total_prompt_tokens, total_completion_tokens = 0, 0

        # 对于每个任务，生成num_samples次
        for _ in range(args.num_samples):
            logger.info("=" * 60)
            logger.info(f"==== rank {rank} begin localizing {instance_id} ====")
            max_attempt_num = args.max_attempt_num
            # 每次生成尝试生成max_attempt_num次
            while max_attempt_num:
                logger.info("=" * 60)
                logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                loc_start_time = time.time()
                try:
                    """
                    Basic instructions:
                        - CodeAct instruction
                        - Few-shot Examples
                    """
                    if args.use_function_calling:
                        system_prompt = function_calling.SYSTEM_PROMPT
                        # system_prompt = CLAUDE_THINKING_INSTRUCTION
                    else:
                        system_prompt = prompt_manager.system_message
                        
                    messages: list[dict] = [{
                        "role": "system",
                        "content": system_prompt
                    }]
                    
                    # 是否使用少样本提示
                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

                    logger.info(f"==== {instance_id} start auto search ====")
                    messages.append({
                        "role": "user",
                        "content": get_task_instruction(bug, include_pr=True, include_hint=True),
                    })
                    # use fork to inherit context!!
                    # 使用fork系统调用创建子进程，子进程会继承父进程的所有资源
                    ctx = mp.get_context('fork')
                    # ctx = mp.get_context('spawn')
                    # 创建一个跨进程共享的队列
                    result_queue = ctx.Manager().Queue()
                    tools = None
                    if args.use_function_calling:
                        tools = function_calling.get_tools(
                            codeact_enable_search_keyword=True,
                            codeact_enable_search_entity=True,
                            # 将代码图的探索去除，做消融实验
                            codeact_enable_tree_structure_traverser=True,
                            codeact_enable_semantic_search=True,
                            simple_desc = args.simple_desc,
                        )
                    process = ctx.Process(target=auto_search_process, kwargs={
                        'result_queue': result_queue,
                        'model_name': args.model,
                        'messages': messages,
                        'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                        'temp': 1,
                        'tools': tools,
                        'use_function_calling': args.use_function_calling,
                        "instance_id": instance_id,
                    })
                    process.start()
                    process.join(timeout=args.timeout)
                    if process.is_alive():
                        logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                        f"reconstruction exceeded timeout. Terminating.")
                        process.terminate()
                        process.join()
                        raise TimeoutError
                    
                    # loc_result, messages, traj_data = result_queue.get()
                    result = result_queue.get()
                    if isinstance(result, dict) and 'error' in result and result['type'] == 'BadRequestError':
                        raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                        # print(f"Error occurred in subprocess: {result['error']}")
                    else:
                        loc_result, messages, traj_data = result
                        
                except litellm.BadRequestError as e:
                    logger.warning(f'{instance_id}----{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue
                except APITimeoutError:
                    logger.warning(f"{instance_id}----APITimeoutError. Try again.")
                    max_attempt_num = max_attempt_num - 1
                    sleep(10)
                    continue
                except TimeoutError:
                    logger.warning(f"{instance_id}----Processing time exceeded 15 minutes. Try again.")
                    max_attempt_num = max_attempt_num - 1
                    continue
                except litellm.exceptions.ContextWindowExceededError as e:
                    logger.warning(f'{instance_id}----{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue

                loc_end_time = time.time()
                if not loc_result:
                    continue # empty result

                total_prompt_tokens += traj_data['usage']['prompt_tokens']
                total_completion_tokens += traj_data['usage']['completion_tokens']
                traj_data['time'] = loc_end_time - loc_start_time
                loc_trajs['trajs'].append(traj_data)

                # generate correct output or finish last attempt
                raw_output_loc.append(loc_result)
                break

        if not raw_output_loc:
            # loc generalization failed
            logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
            loc_res = {
                    "instance_id": instance_id,
                    "found_files": [[]],
                    "found_modules": [[]],
                    "found_entities": [[]],
                    "raw_output_loc": raw_output_loc,
                    "meta_data": {
                        'repo': bug['repo'],
                        'base_commit': bug['base_commit'],
                        'problem_statement': bug['problem_statement'],
                        'patch': bug['patch'],
                        # 'gt_file_changes': gt_file_changes
                    }
                }
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            # all_valid_files = get_all_valid_files()
            all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                instance_id, raw_output_loc
            )
            
            loc_res = {
                "instance_id": instance_id,
                "found_files": all_found_files,
                "found_modules": all_found_modules,
                "found_entities": all_found_entities,
                "raw_output_loc": raw_output_loc,
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug['base_commit'],
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch'],
                    # 'gt_file_changes': gt_file_changes
                }
            }
            
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)
            
            # 模型计算token消耗导致错误，因为litellm可能不支持第三方平台部署的模型
            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                append_to_jsonl(loc_res, traj_file)

        reset_current_issue()


class DummyLock:
    def __enter__(self): return self
    def __exit__(self, *args): pass

def run_localize_single(bug, args, log_queue=None, output_file_lock=None, traj_file_lock=None):
    instance_id = bug["instance_id"]

    # 设置日志器
    logger = logging.getLogger(instance_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 移除旧的 handler 防止重复添加
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # 添加 FileHandler
    file_handler = logging.FileHandler(f"{args.output_folder}/{instance_id}.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"==== Start processing {instance_id} ====")

    prompt_manager = PromptManager(
        prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
        agent_skills_docs=LocationToolsRequirement.documentation,
    )

    set_current_issue(instance_data=bug, rank=0, logger=logger)  # 单进程下 rank 设为 0

    raw_output_loc = []
    loc_trajs = {'trajs': []}
    total_prompt_tokens, total_completion_tokens = 0, 0

    for _ in range(args.num_samples):
        logger.info("=" * 60)
        logger.info(f"==== Begin localizing {instance_id} ====")
        max_attempt_num = args.max_attempt_num

        while max_attempt_num > 0:
            logger.info(f"==== Attempt {max_attempt_num} ====")
            loc_start_time = time.time()

            try:
                if args.use_function_calling:
                    system_prompt = function_calling.SYSTEM_PROMPT
                else:
                    system_prompt = prompt_manager.system_message

                messages = [{"role": "system", "content": system_prompt}]
                logger.info(f"系统提示词:\n {system_prompt}")
                if args.use_example:
                    messages.append({"role": "user", "content": prompt_manager.initial_user_message})
                    logger.info(f"用户提示词:\n {prompt_manager.initial_user_message}")
                logger.info("==== Start auto search ====")
                messages.append({
                    "role": "user",
                    "content": get_task_instruction(bug, include_pr=True, include_hint=True),
                })
                logger.info(f"任务指令:\n {get_task_instruction(bug, include_pr=True, include_hint=True)}")

                result_queue = Queue()
                tools = function_calling.get_tools(
                    codeact_enable_search_keyword=True,
                    codeact_enable_search_entity=True,
                    codeact_enable_tree_structure_traverser=True,
                    codeact_enable_semantic_search=False,
                    simple_desc = args.simple_desc,
                    thought_tool=True,
                ) if args.use_function_calling else None
                logger.info(f"是否工具调用：{args.use_function_calling}")
                logger.info(f"工具：\n{tools}")

                auto_search_process(
                    result_queue=result_queue,
                    model_name=args.model,
                    messages=messages,
                    fake_user_msg=auto_search.FAKE_USER_MSG_FOR_LOC,
                    tools=tools,
                    use_function_calling=args.use_function_calling,
                    instance_id=instance_id,
                    logger=logger,
                    temp=0.7,
                    bug=bug,
                    critic_agent=args.critic_agent,
                    reflection_agent=args.reflection_agent,
                )
                result = result_queue.get_nowait()
                logger.info(f"==== auto search succeed ====")
                # logger.info(f"{result}")
                if isinstance(result, dict) and 'error' in result:
                    raise Exception(result['error'])

                loc_result, messages, traj_data = result

            except Exception as e:
                logger.warning(f"Error occurred: {e}. Retrying...")
                max_attempt_num -= 1
                continue

            loc_end_time = time.time()
            if not loc_result:
                logger.info("Empty result.")
                max_attempt_num = max_attempt_num - 1
                continue

            total_prompt_tokens += traj_data['usage']['prompt_tokens']
            total_completion_tokens += traj_data['usage']['completion_tokens']
            traj_data['time'] = loc_end_time - loc_start_time
            loc_trajs['trajs'].append(traj_data)
            raw_output_loc.append(loc_result)
            break

    # Save results
    if not raw_output_loc:
        logger.info(f"==== {instance_id} failed ====")
        loc_res = {
            "instance_id": instance_id,
            "found_files": [[]],
            "found_modules": [[]],
            "found_entities": [[]],
            "raw_output_loc": raw_output_loc,
            "meta_data": {
                'repo': bug['repo'],
                'base_commit': bug['base_commit'],
                'problem_statement': bug['problem_statement'],
                'patch': bug['patch'],
                # 'gt_file_changes': gt_file_changes
            }
        }
        with output_file_lock:
            append_to_jsonl(loc_res, args.output_file)
    else:
        logger.info(f"==== {instance_id} succeed ====")
        logger.info(f"raw_ouput: \n{raw_output_loc}")
        all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
            instance_id, raw_output_loc
        )

        logger.info(f"解析后: \n{all_found_files}\n====\n{all_found_modules}\n====\n{all_found_entities}")
        loc_res = {
            "instance_id": instance_id,
            "found_files": all_found_files,
            "found_modules": all_found_modules,
            "found_entities": all_found_entities,
            "raw_output_loc": raw_output_loc,
            "meta_data": {
                'repo': bug['repo'],
                'base_commit': bug['base_commit'],
                'problem_statement': bug['problem_statement'],
                'patch': bug['patch'],
                # 'gt_file_changes': gt_file_changes
            }
        }
        with output_file_lock:
            append_to_jsonl(loc_res, args.output_file)

        cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
        loc_res['usage'] = {
            'cost($)': round(cost, 5),
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens
        }
        loc_res['loc_trajs'] = loc_trajs
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        with traj_file_lock:
            append_to_jsonl(loc_res, traj_file)

    reset_current_issue()


def localize(args):
    # 从Huggingface Hub中加载数据集
    bench_data = load_dataset(args.dataset, split=args.split)
    bench_tests = filter_dataset(bench_data, 'instance_id', args.used_list)
    if args.eval_n_limit:
        # 限制使用的数据集的样本数量，选择前eval_n_limit个样本
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        # bench_tests = bench_tests.select(range(0, eval_n_limit))
        # 这里改为随机从数据集中随机选择eval_n_limit个样本
        bench_tests_shuffled = bench_tests.shuffle(seed=42)
        bench_tests = bench_tests_shuffled.select(range(0, eval_n_limit))

        logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')
    # 创建一个Manager对象，用于在多线程中管理共享数据，后续可以通过这个对象创建共享队列、字典等
    # manager = mp.Manager()
    # 创建一个共享的队列，用于在多个进程之间安全地传递数据
    # queue = manager.Queue()
    # 获取锁，用于控制对共享数据的访问
    # output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()

    # collect processed instances
    processed_instance = []
    if os.path.exists(args.output_file):
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        locs = load_jsonl(args.output_file)        
        if args.rerun_empty_location:
            # 如果设置了参数对空结果重新运行，则将旧结果备份，同时筛选出不为空的结果
            traj_datas = load_jsonl(traj_file)
            backup_loc_output = backup_file(args.output_file)
            backup_traj_output = backup_file(traj_file)
            clear_file(args.output_file)
            clear_file(traj_file)
            for loc in locs:
                if loc['found_files'] != [[]]:
                    append_to_jsonl(loc, args.output_file)
                    processed_instance.append(loc['instance_id'])
                    
            for loc_traj in traj_datas:
                if loc_traj['found_files'] != [[]]:
                    append_to_jsonl(loc_traj, traj_file)
        else:
            processed_instance = [loc['instance_id'] for loc in locs]

    # 如果Bug没有被处理过则加入到队列中，同时记录需要处理的bug数量
    num_bugs = 0
    bugs_to_process = []
    for bug in bench_tests:
        instance_id = bug["instance_id"]
        # 测试使用
        # 在sparse上无法预测正确的数据上做实验
        # if instance_id not in ['sphinx-doc__sphinx-8474', 'astropy__astropy-12907']:
        #     continue

        # 在稀疏分层检索和混合检索上都无法定位成功的数据上进行实验
        # deepseekv3.1文件定位错误结果
        # if instance_id not in ['django__django-14997', 'sympy__sympy-11870', 'sympy__sympy-12236', 'sympy__sympy-13031'
        #                        , 'sympy__sympy-13146', 'sympy__sympy-13895', 'sympy__sympy-17630', 'sympy__sympy-18087'
        #                        , 'sympy__sympy-20322', 'sympy__sympy-21379', 'sympy__sympy-21612', 'sympy__sympy-21627']:
        #     continue
        #
        # if instance_id not in ['django__django-11630', 'django__django-11797', 'django__django-11910', 'django__django-11999'
        #                        , 'django__django-12113', 'django__django-12470', 'django__django-12856', 'django__django-13158'
        #                        , 'django__django-14580', 'django__django-15388', 'pylint-dev__pylint-7114', 'pylint-dev__pylint-7228'
        #                        , 'pytest-dev__pytest-8906', 'pytest-dev__pytest-9359', 'sphinx-doc__sphinx-8801']:
        #      continue
        # if not instance_id in ['django__django-11910', 'django__django-12184',
        #                        'django__django-15738', 'pytest-dev__pytest-7220']:
        #     continue

        # if not instance_id in ['django__django-15388', 'django__django-13033', 'scikit-learn__scikit-learn-11281',
        #                        'pytest-dev__pytest-5495', 'pytest-dev__pytest-9359', 'scikit-learn__scikit-learn-25500',
        #                        'sphinx-doc__sphinx-7738', 'sphinx-doc__sphinx-8595', 'matplotlib__matplotlib-22711',
        #                        'matplotlib__matplotlib-23476', 'matplotlib__matplotlib-23562', 'atplotlib__matplotlib-23913',
        #                        'matplotlib__matplotlib-25311', 'pallets__flask-4045', 'pydata__xarray-4493', 'django__django-16229']:
        #      continue


        if instance_id in processed_instance:
            print(f"instance {instance_id} has already been processed, skip.")
        else:
            # queue.put(bug)
            bugs_to_process.append(bug)
            num_bugs += 1

    # 设置一个多线程环境下的日志处理机制，创建一个共享队列，在多线程环境中安全传递日志消息
    # log_queue = manager.Queue()
    log_queue = None
    output_file_lock = DummyLock()
    traj_file_lock = DummyLock()

    for idx, bug in enumerate(bugs_to_process):
        print(f"{idx + 1}/{num_bugs} Processing: {bug['instance_id']}")
        run_localize_single(bug, args, log_queue, output_file_lock, traj_file_lock)
    # queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    # queue_listener.start()
    # 使用torch.multiprocessing.spawn创建多个进程，每个进程负责处理一个bug
    # 多个线程并行执行run_localize函数，nprocs指定启动的线程数量，join=True表示主进程等待所有子进程结束
    # mp.spawn(
    #     run_localize,
    #     nprocs=min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs,
    #     args=(args, queue, log_queue, output_file_lock, traj_file_lock),
    #     join=True
    # )
    # queue_listener.stop()
    
    if args.rerun_empty_location:
        try:
            delete_file(backup_loc_output)
            delete_file(backup_traj_output)
        except:
            return


def merge(args, bench_data=None):
    args.merge_file = os.path.join(args.output_folder, 'merged_' + os.path.basename(args.output_file))

    if args.ranking_method == 'mrr':
        args.merge_file = args.merge_file.replace('.jsonl', f'_{args.ranking_method}.jsonl')

    clear_file(args.merge_file)

    # 用于去重的集合和记录重复id的列表
    seen_instance_ids = set()
    duplicate_instance_ids = []

    with open(args.output_file, 'r') as file:
        cnt = 1
        for line in file:
            loc_data = json.loads(line)
            instance_id = loc_data['instance_id']

            # 检查是否重复
            if instance_id in seen_instance_ids:
                duplicate_instance_ids.append(instance_id)
                logging.info(f"跳过重复的 instance_id: {instance_id}")
                continue

            # 标记为已处理
            seen_instance_ids.add(instance_id)

            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_modules'] = []
                loc_data['found_entities'] = []
            else:
                loc_data['found_files'] = loc_data['found_files']
                loc_data['found_modules'] = loc_data['found_modules']
                loc_data['found_entities'] = loc_data['found_entities']
                ranked_files, ranked_modules, ranked_funcs = merge_sample_locations(loc_data['found_files'],
                                                                                    loc_data['found_modules'],
                                                                                    loc_data['found_entities'],
                                                                                    ranking_method=args.ranking_method,
                                                                                    )
                loc_data['found_files'] = ranked_files
                loc_data['found_modules'] = ranked_modules
                loc_data['found_entities'] = ranked_funcs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')

    # 输出去重信息
    if duplicate_instance_ids:
        logging.info(f"=" * 60)
        logging.info(f"去重完成，共去除 {len(duplicate_instance_ids)} 个重复的 instance_id:")
        for dup_id in duplicate_instance_ids:
            logging.info(f"  - {dup_id}")
        logging.info(f"=" * 60)
    else:
        logging.info("没有发现重复的 instance_id")

    logging.info(f"最终保留 {len(seen_instance_ids)} 个唯一的 instance_id")


def main():
    parser = argparse.ArgumentParser()
    # action表示如果提供了命令行参数--localize，则该参数得值会被设置为True, 否则为False
    parser.add_argument("--localize", default=True)
    parser.add_argument("--merge", default=True)
    parser.add_argument("--use_example", default=False)
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_n_limit", type=int, default=274)
    parser.add_argument("--used_list", type=str, default='selected_ids')
    # 为了适配windows系统，使用反斜杠
    parser.add_argument("--output_folder", type=str, default="lite_raw_res")
    parser.add_argument("--output_file", type=str, default="res.jsonl")
    parser.add_argument("--merge_file", type=str, default="merged_res.jsonl")

    # 加参数，用于消融实验
    # 是否用批判智能体
    parser.add_argument("--critic_agent", default=True)
    # 是否用反省agent进行经验总结
    parser.add_argument("--reflection_agent", default=True)

    parser.add_argument(
        "--model", type=str,
        # default="deepseek/deepseek-chat",
        # default="volcengine/deepseek-v3-250324",
        default="volcengine/deepseek-v3-2-251201",
        choices=["gpt-4o", 
                 "azure/gpt-4o", "openai/gpt-4o-2024-05-13",
                 "deepseek/deepseek-chat", "deepseek-ai/DeepSeek-R1",
                 "litellm_proxy/claude-3-5-sonnet-20241022", "litellm_proxy/gpt-4o-2024-05-13", "litellm_proxy/o3-mini-2025-01-31",
                 # fine-tuned model
                 "openai/qwen-7B", "openai/qwen-7B-128k", "openai/ft-qwen-7B", "openai/ft-qwen-7B-128k",
                 "openai/qwen-32B", "openai/qwen-32B-128k", "openai/ft-qwen-32B", "openai/ft-qwen-32B-128k",
                 "openai/deepseek-v3-250324", "openai/qwen3-235b-a22b", "openai/qwen-plus", "volcengine/deepseek-v3-250324",
                 "volcengine/deepseek-v3-2-251201"
        ]
    )
    parser.add_argument("--use_function_calling", default=True,
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    parser.add_argument("--simple_desc", default=False,
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    parser.add_argument("--max_attempt_num", type=int, default=1,
                        help='Only use in generating training trajectories.')
    parser.add_argument("--num_samples", type=int, default=2)
    # parser.add_argument("--num_processes", type=int, default=5)
    
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rerun_empty_location", default=False)
    args = parser.parse_args()
    # repo_ops里面需要修改search_code_snippets函数，因为之前是使用sparse检索，现在改为dense检索

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        # vars在python中用于获取对象的属性和属性值，返回包含对象所有属性和值的字典
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )

    # if args.localize:
    #     localize(args)

    # bench_data = load_dataset(args.dataset, split=args.split)
    # 筛选出前40个数据
    # bench_data = bench_data[:args.eval_n_limit]
    if args.merge:
        # merge(args, bench_data=bench_data)
        merge(args)

if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
