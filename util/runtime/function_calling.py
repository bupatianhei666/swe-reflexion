
"""This file contains the function calling implementation for different actions.
"""

import json
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)
from util.actions.action import (
    Action,
    FinishAction,
    # CmdRunAction,
    IPythonRunCellAction,
    MessageAction, ThoughtAction,
)
from .finish import FinishTool, ThoughtTool
from .structure_tools import ExploreTreeStructure, ExploreTreeStructure_simple
from .content_tools import SearchEntityTool, SearchRepoTool, SemanticSearchTool
import logging
logger = logging.getLogger()

ALL_FUNCTIONS = ['explore_tree_structure', 'keyword_search_code_snippets', 'get_entity_contents', 'semantic_search_code_snippets']
# ALL_FUNCTIONS = ['explore_tree_structure', 'search_code_snippets', 'get_entity_contents']
SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>
"""


def combine_thought(action: Action, thought: str) -> Action:
    if hasattr(action, 'raw_content') and thought:
        action.raw_content = thought
    if hasattr(action, 'thought') and thought:
        action.thought += '\n' + thought
    return action


def response_to_actions(response: ModelResponse) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    assistant_msg = response.choices[0].message
    if assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e
            if tool_call.function.name == 'finish':
                if list(arguments.values()):
                    # 这里可能存在错误
                    """
                    Message(content='', role='assistant', tool_calls=[ChatCompletionMessageToolCall(function=Function(arguments='{"locations": ["src/_pytest/junitxml.py:LogXML", "src/_pytest/junitxml.py:_NodeReporter"]}', name='finish'), id='call_8cxmuroqceirammjadn86i1h', type='function')], function_call=None, provider_specific_fields={'refusal': None})
                    """
                    action = FinishAction(thought=list(arguments.values())[0])
                else:
                    action = FinishAction()

            # 新增thought tool解析
            elif tool_call.function.name == 'save_thought':
                if list(arguments.values()):
                    action = ThoughtAction(thought=list(arguments.values())[0], tool_call_id=tool_call.id)
                else:
                    action = ThoughtAction()
                
            elif tool_call.function.name in ALL_FUNCTIONS:
                # We implement this in agent_skills, which can be used via Jupyter
                func_name = tool_call.function.name
                code = f'print({func_name}(**{arguments}))'
                logger.info(f'TOOL CALL: {func_name} with code: {code}')
                action = IPythonRunCellAction(code=code,
                                              function_name=func_name,
                                              tool_call_id=tool_call.id)  # include_extra=False
            else:
                raise RuntimeError(f'Unknown tool call: {tool_call.function.name}')

            # We only add thought to the first action
            if i == 0 and not isinstance(action, ThoughtAction):
                action = combine_thought(action, thought)

            actions.append(action)
    else:
        actions.append(
            MessageAction(raw_content=assistant_msg.content, content=assistant_msg.content)
        )

    assert len(actions) >= 1
    return actions


def get_tools(
        codeact_enable_search_keyword: bool = False,
        codeact_enable_search_entity: bool = False,
        codeact_enable_tree_structure_traverser: bool = False,
        codeact_enable_semantic_search: bool = False,
        simple_desc: bool = False,
        thought_tool: bool = False,
) -> list[ChatCompletionToolParam]:
    # 新加入thought tool工具
    tools = [FinishTool]
    if thought_tool:
        tools.append(ThoughtTool)
    # if codeact_enable_cmd:
    #     tools.append(CmdRunTool)
    # 这里换一下位置，进行实验
    if codeact_enable_tree_structure_traverser:
        if simple_desc:
            tools.append(ExploreTreeStructure_simple)
        else:
            tools.append(ExploreTreeStructure)
    if codeact_enable_search_keyword:
        tools.append(SearchRepoTool)
    if codeact_enable_search_entity:
        tools.append(SearchEntityTool)
    if codeact_enable_semantic_search:
        tools.append(SemanticSearchTool)

    return tools


