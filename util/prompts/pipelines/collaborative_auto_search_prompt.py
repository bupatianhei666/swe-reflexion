TASK_INSTRUECTION="""
Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.

** IMPORTANT: Collaborative Task & Dynamic Feedback **
You are working as part of a collaborative team of agents to solve this issue. During your process, you may receive feedback from other agents about exploration paths they have already tried.
This feedback is not an instruction to discard your work. Your exploration history up to the current point remains valid. Instead, treat this feedback as a critical lesson for your next decision.
When you receive such feedback, you must:
1. Compare:Carefully compare the ineffective path from the feedback with your own current tool-call history.
2. Analyze: If your current path matches the initial steps of the ineffective one, recognize that the subsequent steps taken by the other agent led to a dead end.
3. Adjust: You must adjust your immediate next step to avoid repeating the same mistake. Use the knowledge that a specific follow-up path is incorrect to make a more informed choice for your next action, selecting an alternative tool or pursuing a different line of inquiry from your current, valid position.

Follow these steps to localize the issue:
## Step 1: Categorize and Extract Key Problem Information
 - Classify the problem statement into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Identify modules (files, classes, functions) in the '{package_name}' package mentioned in each category.
 - Use these extracted keywords and any mentioned line numbers as initial points for your investigation.
   You can use `keyword_search_code_snippets` with these keywords or line numbers to get an initial understanding.
   Example: `keyword_search_code_snippets(search_terms=["RelevantKeyword"], file_path_or_pattern="path/to/file.py")`
   Example: `keyword_search_code_snippets(line_nums=[123], file_path_or_pattern="path/to/specific_file.py")`

## Step 2: Locate Referenced and Initial Candidate Modules
- Based on Step 1, accurately determine specific modules (files, classes, functions) that are explicitly mentioned or strongly implied by the issue description.
- For each initially identified entity, you might want to retrieve its full content if it seems central to the issue.
  Use `get_entity_contents(entity_names=["path/to/file.py:ClassName.method_name"])` to get the full code of a specific entity.
  Use `keyword_search_code_snippets(search_terms=["path/to/file.py:ClassName.method_name"])` for a preview if you are unsure.
- Pay special attention to distinguishing between modules with similar names using the context from the issue and the code structure.
- Output Format for collected relevant modules (keep a running list in your thoughts):
    - Use the format: 'file_path:QualifiedName'
    - E.g., for a function `calculate_sum` in the `MathUtils` class located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.

## Step 3: Iterative Analysis: Explore Dependencies and Reconstruct Execution Flow
Your goal is to understand how the initially identified entities connect to the issue and to trace the potential execution path leading to the problem. This often requires iterative exploration.

1.  **Initial Code Examination & Hypothesis Formation**:
    * Review the code of entities identified in Step 2 (obtained via `get_entity_contents` or `keyword_search_code_snippets`).
    * Formulate hypotheses about how these entities might be related to the reported issue. Does the issue stem from internal logic, or interactions with other components?

2.  **Exploring Relationships with `explore_tree_structure` (Corresponds to TraverseGraph)**:
    * **When to use**:
        * If the initial code snippets are insufficient to understand the problem.
        * If the issue description or your analysis hints at interactions between multiple components, some of which may not be explicitly named.
        * If you need to understand callers (upstream) or callees (downstream) of a function/method.
        * If you need to explore the inheritance hierarchy of a class or the import relationships of a file/module.
        * If the root cause of the issue is not obvious from directly examining the initially identified entities.
    * **How to use**:
        * `start_entities`: Provide a list of entity IDs (e.g., from Step 2 or previous search/exploration steps).
        * `direction`:
            * `'downstream'`: To see what other code the `start_entities` call or affect (e.g., functions called, classes instantiated).
            * `'upstream'`: To see what code calls or affects the `start_entities` (e.g., where a function is called from, parent classes).
            * `'both'`: If the relationship is unclear or you need a comprehensive view.
        * `traversal_depth`: Start with a small depth (e.g., 1 or 2). Increase if more context is needed but be mindful of output size.
        * `dependency_type_filter`:
            * To trace execution flow: `['invokes', 'imports']`.
            * To understand class structure: `['inherits', 'contains']`.
            * To explore file/directory structure or general imports: `['contains', 'imports']`.
            * Leave empty or use `None` to see all types of dependencies if unsure.
    * **Example**: `explore_tree_structure(start_entities=["path/to/file.py:ClassName.method_name"], direction="upstream", traversal_depth=1, dependency_type_filter=["invokes"])`

3.  **Analyzing Exploration Results and Deepening Understanding**:
    * The output of `explore_tree_structure` will be a graph showing related entities.
    * Identify new, potentially relevant entities from this graph.
    * For these newly discovered entities:
        * Use `get_entity_contents(entity_names=["newly/found/entity_id"])` to get their full code for a detailed understanding.
        * Alternatively, use `keyword_search_code_snippets(search_terms=["newly/found/entity_id"])` for a quick preview.

4.  **Refining Search with Keywords**:
    * If new keywords, error messages, or specific variable/function names emerge during your analysis of code (from `get_entity_contents` or `keyword_search_code_snippets`) or graph structures (from `explore_tree_structure`), use these new terms with `keyword_search_code_snippets` to find other potentially relevant code regions across the codebase.
    * Example: `keyword_search_code_snippets(search_terms=["new_keyword_discovered"])`

5.  **Iterate**:
    * Repeat steps 1-4. If a newly found entity seems critical but its role or connections are still unclear, use it as a `start_entity` for another `explore_tree_structure` call.
    * The goal is to build a clear understanding of the "fault to failure" path.

6.  **Reconstruct the Execution Flow**:
    * Based on all gathered information, identify the main entry point(s) that trigger the problem.
    * Trace the sequence of function calls, class interactions, and data flow relevant to the issue.
    * Pinpoint where the code's behavior likely diverges from expectations, leading to the reported problem.
    * Focus on the problematic flow, avoiding irrelevant details.

## Step 4: Locate Specific Areas for Modification
- Based on your reconstructed execution flow and understanding of the involved components, pinpoint the specific files, functions, classes, or even precise line numbers that most likely need modification to resolve the issue.
- Consider upstream and downstream dependencies: how will changes in one place affect others? Are there shared utilities that need careful handling?
- If applicable (e.g., for feature requests), identify where to introduce new fields, functions, or variables.
- Think thoroughly: List multiple potential solution approaches or areas if the exact point is ambiguous, but prioritize the most likely ones.

** IMPORTANT ** 
- After each tool call step, you should conduct a detailed analysis of the result and make the correct decision for the next step.
- Do not call the "finish" tool to output results until you have completed the reasoning plan or carefully understood the dependencies between codes and obtained the correct code localization result.

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.
Your answer would better include about 5 files.

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

Return just the location(s)

Note: Your thinking should be thorough and so it's fine if it's very long.
"""

# SEMANTIC_TASK_DESCRIPTION = """
# Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.
#
# ** IMPORTANT: Collaborative Task & Dynamic Feedback **
# You are working as part of a collaborative team of agents to solve this issue. During your process, you may receive feedback from other agents about exploration paths they have already tried.
# This feedback is not an instruction to discard your work. Your exploration history up to the current point remains valid. Instead, treat this feedback as a critical lesson for your next decision.
# When you receive such feedback, you must:
# 1. Compare: Carefully compare the ineffective path from the feedback with your own current tool-call history.
# 2. Analyze: If your current path matches the initial steps of the ineffective one, recognize that the subsequent steps taken by the other agent led to a dead end.
# 3. Adjust: You must adjust your immediate next step to avoid repeating the same mistake. Use the knowledge that a specific follow-up path is incorrect to make a more informed choice for your next action, selecting an alternative tool or pursuing a different line of inquiry from your current, valid position.
#
# ** IMPORTANT: Search Tool Selection Strategy **
# You have TWO complementary search tools available. Choose the appropriate tool based on the nature of your query:
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                         SEARCH TOOL DECISION GUIDE                          │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  USE `keyword_search_code_snippets` WHEN:                                   │
# │  ├── You have EXACT names: class names, function names, variable names     │
# │  ├── You have specific error messages or exception names                   │
# │  ├── You have file paths or module names mentioned in the issue            │
# │  ├── You need to find code at specific line numbers                        │
# │  └── Example: keyword_search_code_snippets(search_terms=["ValidationError"])│
# │                                                                             │
# │  USE `semantic_search_code_snippets` WHEN:                                  │
# │  ├── The issue describes BEHAVIOR or FUNCTIONALITY without exact names     │
# │  ├── You need to find code that "does something" conceptually              │
# │  ├── Entity names in the issue are vague, generic, or potentially wrong    │
# │  ├── You want to find code handling a specific type of logic or operation  │
# │  └── Example: semantic_search_code_snippets(query="validates user input    │
# │               and raises appropriate errors for invalid data formats")     │
# │                                                                             │
# │  COMBINE BOTH TOOLS WHEN:                                                   │
# │  ├── Issue has partial information (some names + behavioral descriptions)  │
# │  ├── Initial keyword search returns no results or irrelevant results       │
# │  ├── You need to verify semantic search results with precise lookups       │
# │  └── Strategy: semantic_search → identify area → keyword_search → pinpoint │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Follow these steps to localize the issue:
#
# ## Step 1: Analyze Problem Statement and Determine Search Strategy
#
# ### 1.1 Categorize the Problem Information
# Classify the problem statement into the following categories:
# - **Problem description**: What is the reported issue or unexpected behavior?
# - **Error trace**: Any stack traces, error messages, or exceptions?
# - **Code to reproduce**: Any code snippets demonstrating the issue?
# - **Additional context**: Environment details, version info, related issues?
#
# ### 1.2 Extract Search Targets (CRITICAL: Distinguish Between Explicit and Implicit Information)
#
# **A. Explicit Information (Use Keyword Search):**
# - Exact file paths, class names, function names, variable names
# - Specific error messages or exception types
# - Line numbers mentioned in stack traces
# - Module or package names
#
# **B. Implicit/Behavioral Information (Use Semantic Search):**
# - Descriptions of what the code should do vs. what it actually does
# - Functional requirements or expected behaviors
# - Logical operations or data transformations mentioned
# - User workflows or feature descriptions
#
# ### 1.3 Execute Initial Search Based on Information Type
#
# **If explicit entity names are available:**
# ```
# keyword_search_code_snippets(search_terms=["ExactClassName", "exact_function_name"])
# keyword_search_code_snippets(line_nums=[123], file_path_or_pattern="path/to/file.py")
# ```
#
# **If only behavioral descriptions are available (common in unclear issues):**
# ```
# semantic_search_code_snippets(query="describe the functionality you're looking for in natural language")
# ```
#
# **Example Transformation (Behavioral → Semantic Query):**
# - Issue says: "The validation doesn't work properly when users submit forms"
# - Semantic query: "form validation logic that checks user input and handles submission errors"
#
# - Issue says: "Data is not being saved correctly to the database"
# - Semantic query: "database save operation that persists data and handles transaction commits"
#
# ## Step 2: Locate Candidate Modules Through Adaptive Search
#
# ### 2.1 Primary Search Execution
#
# **Scenario A: Rich Explicit Information**
# - Directly search for mentioned entities using `keyword_search_code_snippets`
# - Verify found entities match the context described in the issue
#
# **Scenario B: Limited or No Explicit Information (Fuzzy Issues)**
# - Analyze the PROBLEM BEHAVIOR described in the issue
# - Formulate semantic queries that capture:
#   * What functionality is affected?
#   * What operation is failing?
#   * What data or workflow is involved?
# - Execute `semantic_search_code_snippets` with these behavioral descriptions
# - Example:
#   ```
#   # Issue: "Labels are not applied properly during training"
#   # No specific class/function names given
#
#   semantic_search_code_snippets(
#       query="training label application logic that assigns labels to training data during model training process"
#   )
#   ```
#
# **Scenario C: Mixed Information**
# - Start with keyword search for any explicit names
# - Use semantic search to find related components not explicitly mentioned
# - Cross-reference results to build a complete picture
#
# ### 2.2 Validate and Expand Search Results
#
# For each search result (from either tool):
# - Use `get_entity_contents(entity_names=["path/to/file.py:ClassName.method_name"])` to examine full code
# - Assess relevance: Does this code relate to the described problem?
# - If relevant, add to your candidate list; if not, refine your search query
#
# **Output Format for collected relevant modules (maintain in your thoughts):**
# - Use the format: `file_path:QualifiedName`
# - E.g., `src/helpers/math_helpers.py:MathUtils.calculate_sum`
#
# ## Step 3: Iterative Analysis - Explore Dependencies and Reconstruct Execution Flow
#
# ### 3.1 Initial Code Examination & Hypothesis Formation
# - Review code of entities identified in Step 2
# - Formulate hypotheses about how these entities relate to the reported issue
# - Consider: Is the issue in internal logic, or in interactions with other components?
#
# ### 3.2 Exploring Relationships with `explore_tree_structure`
#
# **When to use:**
# - Initial code examination is insufficient to understand the problem
# - Issue hints at interactions between multiple components
# - You need to understand callers (upstream) or callees (downstream)
# - Root cause is not obvious from directly examining identified entities
#
# **How to use:**
# - `start_entities`: Entity IDs from your candidate list
# - `direction`:
#   * `'downstream'`: See what code the entities call or affect
#   * `'upstream'`: See what code calls or affects the entities
#   * `'both'`: For comprehensive view when relationship is unclear
# - `traversal_depth`: Start with 1-2, increase if needed
# - `dependency_type_filter`:
#   * Execution flow: `['invokes', 'imports']`
#   * Class structure: `['inherits', 'contains']`
#   * File structure: `['contains', 'imports']`
#
# **Example:**
# ```
# explore_tree_structure(
#     start_entities=["path/to/file.py:ClassName.method_name"],
#     direction="upstream",
#     traversal_depth=1,
#     dependency_type_filter=["invokes"]
# )
# ```
#
# ### 3.3 Analyzing Results and Deepening Understanding
#
# For newly discovered entities from graph exploration:
# - Get full code: `get_entity_contents(entity_names=["newly/found/entity_id"])`
# - Quick preview: `keyword_search_code_snippets(search_terms=["newly/found/entity_id"])`
#
# ### 3.4 Refining Search with Discovered Information
#
# **As you analyze code, you may discover new search targets:**
#
# - **New explicit names found** → Use `keyword_search_code_snippets`
#   ```
#   # Found a function name "process_labels" in the code
#   keyword_search_code_snippets(search_terms=["process_labels"])
#   ```
#
# - **New behavioral patterns identified** → Use `semantic_search_code_snippets`
#   ```
#   # Discovered the issue might be in augmentation logic
#   semantic_search_code_snippets(query="image augmentation that transforms and crops training data")
#   ```
#
# ### 3.5 Iterate Until Clear Understanding
#
# Repeat Steps 3.1-3.4 until you can:
# - Identify main entry point(s) that trigger the problem
# - Trace the sequence of function calls and data flow
# - Pinpoint where code behavior diverges from expectations
#
# ## Step 4: Locate Specific Areas for Modification
#
# ### 4.1 Synthesize Findings
# Based on your analysis:
# - Identify the root cause location(s)
# - Consider upstream/downstream effects of potential changes
# - Identify shared utilities that need careful handling
#
# ### 4.2 Prioritize Locations
# - Primary: Most likely root cause locations
# - Secondary: Related code that may need consistent updates
# - Tertiary: Test files or documentation that may need updates
#
# ### 4.3 Verify Completeness
# Before finalizing:
# - Have you traced the full execution path?
# - Are there edge cases or related code paths not yet examined?
# - If unsure, use additional semantic searches to verify no important areas were missed
#
# ** IMPORTANT REMINDERS **
# - After each tool call, analyze results thoroughly before deciding next steps
# - Do NOT call "finish" until you have a clear understanding of the code flow and confident localization
# - When keyword search fails, try semantic search with behavioral descriptions
# - When semantic search returns too many results, refine with more specific queries or use keyword search to filter
#
# ## Output Format for Final Results:
# Your final output should list the locations requiring modification, wrapped with triple backticks ```
# Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.
# Your answer should include about 5 files.
#
# ### Examples:
# ```
# full_path1/file1.py
# line: 10
# class: MyClass1
# function: my_function1
#
# full_path2/file2.py
# line: 76
# function: MyClass2.my_function2
#
# full_path3/file3.py
# line: 24
# line: 156
# function: my_function3
# ```
#
# Return just the location(s)
#
# Note: Your thinking should be thorough and so it's fine if it's very long.
# """


SEMANTIC_TASK_DESCRIPTION = """
Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.

** IMPORTANT: Collaborative Task & Dynamic Feedback **
You are working as part of a collaborative team of agents to solve this issue. During your process, you may receive feedback from other agents about exploration paths they have already tried.
This feedback is not an instruction to discard your work. Your exploration history up to the current point remains valid. Instead, treat this feedback as a critical lesson for your next decision.
When you receive such feedback, you must:
1. Compare: Carefully compare the ineffective path from the feedback with your own current tool-call history.
2. Analyze: If your current path matches the initial steps of the ineffective one, recognize that the subsequent steps taken by the other agent led to a dead end.
3. Adjust: You must adjust your immediate next step to avoid repeating the same mistake. Use the knowledge that a specific follow-up path is incorrect to make a more informed choice for your next action, selecting an alternative tool or pursuing a different line of inquiry from your current, valid position.

** IMPORTANT: Search Tool Selection Strategy **
You have access to two complementary search tools. Your success depends on choosing the right one based on the information type:
1. `keyword_search_code_snippets`: Use this when you have **EXACT** identifiers (class names, function names, variable names, file paths, specific line numbers, or error message strings).
2. `semantic_search_code_snippets`: Use this when you have **CONCEPTUAL** or **BEHAVIORAL** descriptions (e.g., "functionality that handles user authentication," "code that calculates tax," or vague references like "the validation logic").
3. **Hybrid Approach**: If keyword search fails or returns too many results, use semantic search to narrow down the module, then use keywords or `explore_tree_structure` to pinpoint.

Follow these steps to localize the issue:
## Step 1: Categorize, Extract, and Execute Initial Search
- Classify the problem statement into the following categories:
Problem description, error trace, code to reproduce the bug, and additional context.
- **Decision Point**: Analyze the extracted information to determine your search strategy:
    - **Scenario A (Explicit Identifiers):** If the issue mentions specific modules (files, classes, functions):
        - Use `keyword_search_code_snippets` with these exact terms.
        - Example: `keyword_search_code_snippets(search_terms=["RelevantClassName", "specific_function_name"])`
    - **Scenario B (Behavioral/Implicit):** If the issue describes a bug in behavior (e.g., "data isn't saving correctly") without naming the specific code:
        - Use `semantic_search_code_snippets` with a natural language query describing the functionality.
        - Example: `semantic_search_code_snippets(query="logic that saves data to the database and handles transactions")`

## Step 2: Locate Referenced and Initial Candidate Modules
- Based on Step 1, accurately determine specific modules (files, classes, functions) that are explicitly mentioned or strongly implied by the issue description.
- For each initially identified entity, you might want to retrieve its full content if it seems central to the issue.
    Use `get_entity_contents(entity_names=["path/to/file.py:ClassName.method_name"])` to get the full code of a specific entity.
    Use `keyword_search_code_snippets(search_terms=["path/to/file.py:ClassName.method_name"])` for a preview if you are unsure.
- **Verification**: If you used `semantic_search_code_snippets` in Step 1, you must verify the returned snippets actually match the context of the issue before proceeding.
- Output Format for collected relevant modules (keep a running list in your thoughts):
    - Use the format: 'file_path:QualifiedName'
    - E.g., for a function `calculate_sum` in the `MathUtils` class located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.

## Step 3: Iterative Analysis: Explore Dependencies and Reconstruct Execution Flow
Your goal is to understand how the initially identified entities connect to the issue and to trace the potential execution path leading to the problem. This often requires iterative exploration.

1.  **Initial Code Examination & Hypothesis Formation**:
* Review the code of entities identified in Step 2 (obtained via `get_entity_contents` or `keyword_search_code_snippets`).
* Formulate hypotheses about how these entities might be related to the reported issue. Does the issue stem from internal logic, or interactions with other components?

2.  **Exploring Relationships with `explore_tree_structure` (Corresponds to TraverseGraph)**:
    * **When to use**:
        * If the initial code snippets are insufficient to understand the problem.
        * If the issue description or your analysis hints at interactions between multiple components, some of which may not be explicitly named.
        * If you need to understand callers (upstream) or callees (downstream) of a function/method.
        * If you need to explore the inheritance hierarchy of a class or the import relationships of a file/module.
        * If the root cause of the issue is not obvious from directly examining the initially identified entities.
    * **How to use**:
        * `start_entities`: Provide a list of entity IDs (e.g., from Step 2 or previous search/exploration steps).
        * `direction`:
            * `'downstream'`: To see what other code the `start_entities` call or affect (e.g., functions called, classes instantiated).
            * `'upstream'`: To see what code calls or affects the `start_entities` (e.g., where a function is called from, parent classes).
            * `'both'`: If the relationship is unclear or you need a comprehensive view.
        * `traversal_depth`: Start with a small depth (e.g., 1 or 2). Increase if more context is needed but be mindful of output size.
        * `dependency_type_filter`:
            * To trace execution flow: `['invokes', 'imports']`.
            * To understand class structure: `['inherits', 'contains']`.
            * To explore file/directory structure or general imports: `['contains', 'imports']`.
            * Leave empty or use `None` to see all types of dependencies if unsure.
        * **Example**: `explore_tree_structure(start_entities=["path/to/file.py:ClassName.method_name"], direction="upstream", traversal_depth=1, dependency_type_filter=["invokes"])`

3.  **Analyzing Exploration Results and Deepening Understanding**:
    * The output of `explore_tree_structure` will be a graph showing related entities.
    * Identify new, potentially relevant entities from this graph.
    * For these newly discovered entities:
        * Use `get_entity_contents(entity_names=["newly/found/entity_id"])` to get their full code for a detailed understanding.
        * Alternatively, use `keyword_search_code_snippets(search_terms=["newly/found/entity_id"])` for a quick preview.

4.  **Refining Search with Keywords or Semantics**:
    * If new keywords, error messages, or specific variable/function names emerge during your analysis, use `keyword_search_code_snippets`.
    * **Critical**: If your analysis reveals a "gap" in your understanding where you know *what* needs to happen but not *where* (e.g., "I see where the data is prepped, but I need to find where it is actually written to disk"), use `semantic_search_code_snippets` with a description of that missing step.
    * Example: `semantic_search_code_snippets(query="code that writes the prepared buffer to the disk")`

5.  **Iterate**:
    * Repeat steps 1-4. If a newly found entity seems critical but its role or connections are still unclear, use it as a `start_entity` for another `explore_tree_structure` call.
    * The goal is to build a clear understanding of the "fault to failure" path.

6.  **Reconstruct the Execution Flow**:
    * Based on all gathered information, identify the main entry point(s) that trigger the problem.
    * Trace the sequence of function calls, class interactions, and data flow relevant to the issue.
    * Pinpoint where the code's behavior likely diverges from expectations, leading to the reported problem.
    * Focus on the problematic flow, avoiding irrelevant details.

## Step 4: Locate Specific Areas for Modification
- Based on your reconstructed execution flow and understanding of the involved components, pinpoint the specific files, functions, classes, or even precise line numbers that most likely need modification to resolve the issue.
- Consider upstream and downstream dependencies: how will changes in one place affect others? Are there shared utilities that need careful handling?
- Think thoroughly: List multiple potential solution approaches or areas if the exact point is ambiguous, but prioritize the most likely ones.

** IMPORTANT ** - After each tool call step, you should conduct a detailed analysis of the result and make the correct decision for the next step.
- Do not call the "finish" tool to output results until you have completed the reasoning plan or carefully understood the dependencies between codes and obtained the correct code localization result.

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.
Your answer would better include about 5 files.

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

Return just the location(s)

Note: Your thinking should be thorough and so it's fine if it's very long.
"""