from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_IMPROVED_FINISH_DESCRIPTION = """
Use this tool ONLY as the final step to submit your verified code localization results.

**Invocation Preconditions (MUST be met before use):**
1.  **Completion of Plan:** You MUST have fully executed your entire analysis plan. Do not call this tool if you still have investigation steps pending.
2.  **Verification of All Results:** You MUST have thoroughly verified and cross-referenced all identified code locations. Use evidence from your searches (semantic, keyword) and dependency analysis (`explore_tree_structure`) to confirm that each location is directly relevant to the root cause of the issue.
3.  **High Confidence:** Do not use this tool with speculative or uncorroborated findings. You must have high confidence that the locations you are reporting are correct and sufficient to resolve the GitHub issue. Invoking this tool signifies the definitive conclusion of your analysis.

The output MUST be a string formatted precisely as shown in the example below, detailing each specific file and function.
You MUST output ONLY the localization results.
DO NOT include any additional text before or after the formatted result.

**Output Formatting Rules:**
1.  The entire output string MUST be enclosed in triple backticks (```).
2.  Each distinct code location (typically a function within a file) MUST start with the full file path on a new line.
3.  Following the file path, relevant details for that location (line numbers, class name, function name) MUST each be on their own new line.
4.  A 'function: <FunctionName>' entry is MANDATORY for each reported location. If the function is part of a class, use 'function: <ClassName>.<FunctionName>'.
5.  'line: <number>' entries are highly recommended to pinpoint specific lines within the function. Multiple 'line:' entries are allowed if several lines within the same function are relevant.
6.  'class: <ClassName>' should be included if the function is a method of a class.
7.  Distinct code locations MUST be separated by a single blank line.
8.  Aim to identify approximately 5 key file/function locations, ordered by their importance or likelihood of requiring modification.

**Usage Example (Format to strictly follow for the 'message' parameter):**
finish(message="```\nsrc/module_alpha/file1.py\nline: 42\nline: 45\nclass: AlphaProcessor\nfunction: AlphaProcessor.process_data\n\ncore/utils/helper_functions.py\nline: 101\nfunction: calculate_metrics\n\ntests/unit/test_alpha_processor.py\nline: 30\nclass: TestAlphaProcessor\nfunction: TestAlphaProcessor.test_data_processing_logic\n\nsrc/module_beta/file2.py\nline: 77\nfunction: BetaService.handle_request\n\nconfigs/config_loader.py\nline: 15\nfunction: load_app_configuration\n```")
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_IMPROVED_FINISH_DESCRIPTION, # Use the more detailed description
        parameters={
            'type': 'object',
            'properties': {
                'message': {
                    'type': 'string',
                    'description': (
                        "A precisely formatted multi-line string detailing the localized code locations. "
                        "This string MUST be enclosed in triple backticks (```). "
                        "Each identified location MUST specify the 'file_path' and a 'function: <FunctionName>' (or 'function: <ClassName.FunctionName>'). "
                        "Include 'line: <number>' (one or more) and 'class: <ClassName>' (if applicable) as separate lines under the file path. "
                        "Distinct locations MUST be separated by a blank line. "
                        "For example: \n"
                        "```\n"
                        "full_path1/file1.py\n"
                        "line: 10\n"
                        "class: MyClass1\n"
                        "function: MyClass1.my_function1\n"
                        "\n"
                        "full_path2/file2.py\n"
                        "line: 76\n"
                        "function: my_global_function2\n"
                        "```\n"
                        "Adhere strictly to this structure and the detailed formatting rules provided in the main tool description."
                    )
                }
            },
            'required': ['message']
        }
    ),
)

_THOUGHT_TOOL_DESCRIPTION = """
Use this tool to record your internal monologue, reasoning, and step-by-step plan.
This is for detailing your thinking process BEFORE you are ready to call tools.
Call this tool to explain your analysis of the problem, your strategy for finding code locations,and the conclusions you draw from your investigation. 
This creates a transparent log of your reasoning. 
You can call this tool multiple times to show the evolution of your thought process.
"""
# _THOUGHT_TOOL_DESCRIPTION = ""

thought_tool_parameters = {
    'type': 'object',
    'properties': {
        'message': {
            'type': 'string',
            'description': (
                "A detailed, free-form string capturing your current thoughts. This should articulate "
                "your analysis of the user's request, the conclusions you've drawn, and the plan "
                "you are about to execute. It serves as a log of your internal monologue."
            )
        }
    },
    'required': ['message']
}

ThoughtTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='save_thought',
        description=_THOUGHT_TOOL_DESCRIPTION,
        parameters=thought_tool_parameters
    ),
)