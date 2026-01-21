from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)


_SEARCHENTITY_DESCRIPTION = """
Searches the codebase to retrieve the complete implementations of specified entities based on the provided entity names. 
The tool can handle specific entity queries such as function names, class names, or file paths.

**Usage Example:**
# Search for a specific function implementation
get_entity_contents(['src/my_file.py:MyClass.func_name'])

# Search for a file's complete content
get_entity_contents(['src/my_file.py'])

**Entity Name Format:**
- To specify a function or class, use the format: `file_path:QualifiedName`
  (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum').
- To search for a file's content, use only the file path (e.g., 'src/my_file.py').
"""

SearchEntityTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_entity_contents',
        description=_SEARCHENTITY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'entity_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        "A list of entity names to query. Each entity name can represent a function, class, or file. "
                        "For functions or classes, the format should be 'file_path:QualifiedName' "
                        "(e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum'). "
                        "For files, use just the file path (e.g., 'src/my_file.py')."
                    )
                }
            },
            'required': ['entity_names'],
        },
    ),
)

_SEARCHREPO_DESCRIPTION = """Searches the codebase to retrieve relevant code snippets based on given queries(terms or line numbers).
** Note:
- Either `search_terms` or `line_nums` must be provided to perform a search.
- If `search_terms` are provided, it searches for code snippets based on each term:
- If `line_nums` is provided, it searches for code snippets around the specified lines within the file defined by `file_path_or_pattern`.

** Example Usage:
# Search for code content contain keyword `order`, `bill`
search_code_snippets(search_terms=["order", "bill"])

# Search for a class
search_code_snippets(search_terms=["MyClass"])

# Search for context around specific lines (10 and 15) within a file
search_code_snippets(line_nums=[10, 15], file_path_or_pattern='src/example.py')
"""

SearchRepoTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='keyword_search_code_snippets',
        # name='search_code_snippets',
        description=_SEARCHREPO_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'search_terms': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'A list of names, keywords, or code snippets to search for within the codebase. ' \
                                    'This can include potential function names, class names, or general code fragments. '\
                                    'Either `search_terms` or `line_nums` must be provided to perform a search.'     
                },
                'line_nums': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': 'Specific line numbers to locate code snippets within a specified file. '\
                                    'Must be used alongside a valid `file_path_or_pattern`. '\
                                    'Either `line_nums` or `search_terms` must be provided to perform a search.',
                },
                'file_path_or_pattern': {
                    'type': 'string', 
                    'description': 'A glob pattern or specific file path used to filter search results '\
                                   'to particular files or directories. Defaults to "**/*.py", meaning all Python files are searched by default. ' \
                                   'If `line_nums` are provided, this must specify a specific file path.',
                    'default': '**/*.py',
                },
            },
            'required': [],
        },
    ),
)


_SEMANTIC_SEARCH_DESCRIPTION = """Performs a semantic search to retrieve code snippets that are conceptually similar to a natural language description.
This tool is most effective when you describe what the code does in detail. It relies on vector embeddings, so rich context works better than short keywords.

**When to use this tool:**
* You have a description of a feature or a component's responsibility and want to find the code that implements it.
* You can describe a particular behavior or logical process and need to locate where it's handled in the codebase.
* You are looking for code related to a general programming concept, design pattern, or common task (e.g., "data serialization to JSON," "asynchronous HTTP request handling," "input validation logic").

**Formulating your `query`:**
The `query` parameter should be a clear and concise natural language description of:
    * The code's intended behavior or actions.
    * The responsibility or purpose of a module or function.
    * The specific logic or algorithm you are searching for.

**Example Usage:**

1.  **Searching for specific feature implementation:**
    * To find code that handles user authentication and session management:
        `semantic_search_code_snippets(query="Authenticate users, validate session tokens, and handle logout requests")`

2.  **Locating code based on its functional role:**
    * To find the part of the system responsible for processing payment transactions:
        `semantic_search_code_snippets(query="Process credit card payments, interact with payment gateways, and update billing status")`

3.  **Finding code related to a general programming task or concept:**
    * To find how the application parses and validates configuration files:
        `semantic_search_code_snippets(query="Load application configuration from files or environment variables, parse format, and validate settings")`
"""

SemanticSearchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='semantic_search_code_snippets',
        description=_SEMANTIC_SEARCH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'A natural language description of the functionality or code logic to search for. '
                                   'This tool performs a semantic search to find code snippets that are '
                                   'conceptually similar to the query.'
                },
                'file_path_or_pattern': {
                    'type': 'string',
                    'description': 'Optional. A glob pattern or specific file path used to filter semantic search results '
                                   'to particular files or directories. Defaults to searching all relevant code files if not specified.',
                    'default': '**/*.py',
                },
            },
            'required': ['query'],
        },
    ),
)
