NEW_TASK_INSTRUECTION="""
Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.
Previous explore experiences to solve this issue were deemed inefficient or misguided. You must now use the 'lesson learned' from those attempts to formulate a **new and improved strategy** to locate the bug.

## Explore Experiences ##
This section summarizes previous, failed attempt and the key lesson learned. Your new plan **MUST** take this experience into account to avoid repeating mistakes.
---
{explore_experience}
---

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