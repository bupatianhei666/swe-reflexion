from dataclasses import dataclass
from pydantic import BaseModel, Field

class ActionTypeSchema(BaseModel):

    MESSAGE: str = Field(default='message')
    """Represents a message.
    """

    RUN_IPYTHON: str = Field(default='run_ipython')
    """Runs a IPython cell.
    """

    FINISH: str = Field(default='finish')
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

    # 新增thought tool工具类型
    THOUGHT: str = Field(default='thought')
    """Represents the agent's internal thought process. 
    Use this to articulate your reasoning, analysis of the situation, and plan for the next action. 
    This provides a clear trace of your decision-making process before executing another action.
    """

ActionType = ActionTypeSchema()

@dataclass
class Action:
    raw_content: str = ''

# @dataclass
# class FunctionObject:
#     name: str
#     arguments: list

@dataclass
class MessageAction(Action):
    content: str = ''
    action_type: str = ActionType.MESSAGE
    

@dataclass
class FinishAction(Action):
    thought: str = ''
    action_type: str = ActionType.FINISH
    

@dataclass
class IPythonRunCellAction(Action):
    code: str = ''
    thought: str = ''
    function_name: str = ''
    tool_call_id: str = ''
    action_type: str = ActionType.RUN_IPYTHON
    # functions: list[FunctionObject] = Field(default_factory=list)

@dataclass
class ThoughtAction(Action):
    thought: str = ''
    action_type: str = ActionType.THOUGHT
    tool_call_id: str = ''