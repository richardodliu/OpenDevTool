import re
import ast
import json
import signal
import traceback


from copy import deepcopy
from typing import Dict, Generator, List, Optional, Set, Tuple, Iterable

import os
import gzip
import json

from copy import deepcopy
from typing import Dict, Generator, List, Optional, Set, Tuple

python_pattern = re.compile(r'```python[ \t]*[\r\n]+(.*?)```', re.DOTALL | re.IGNORECASE)

def remove_main(code:str)-> str:
    """Remove the main section from a source code string"""
    # Parse the source code into an AST
    tree = ast.parse(code)

    # Remove the main section
    new_body = []
    for node in tree.body:
        if (isinstance(node, ast.If) and
            isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            isinstance(node.test.ops[0], ast.Eq) and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].s == '__main__'):
            # Skip this node to remove it
                continue
        new_body.append(node)

    new_tree = ast.Module(body=new_body, type_ignores=[])

    # Unparse the modified AST back to source code
    modified_code = ast.unparse(new_tree)

    return modified_code

def is_valid_python_syntax(code: str) -> bool:
    """
    Checks if the given code string is syntactically valid Python.

    Args:
        code: A string containing the Python code to check.

    Returns:
        True if the code is syntactically valid, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError, OverflowError, ValueError):
        return False

def extract_longest_valid_code(text: str) -> str:
    """
    Extracts the longest syntactically valid code snippet from the given text.

    Args:
        text: A string containing the source text to extract code from.

    Returns:
        The longest syntactically valid code snippet found in the text, or None if no valid snippet is found.
    """
    lines = text.splitlines()
    max_valid_lines = 0
    best_snippet = None

    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i:j+1])
            if is_valid_python_syntax(current_snippet):
                valid_line_count = sum(1 for line in lines[i:j+1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    best_snippet = current_snippet

    return best_snippet

def extract_python_code(text: str) -> Optional[str]:
    """
    Extracts Python code from the given text.
    First attempts to find Python code blocks using regex, then falls back
    to extracting the longest valid Python snippet from the entire text.

    Args:
        text: A string containing the source text to extract Python code from.

    Returns:
        The extracted Python code as a string, or None if no valid code is found.
    """
    python_matches = python_pattern.findall(text)

    if python_matches:
        code_snippets = [extract_longest_valid_code(match.strip()) for match in python_matches]
        valid_snippets = [snippet for snippet in code_snippets if snippet]
        return '\n\n'.join(valid_snippets) if valid_snippets else None
    else:
        return extract_longest_valid_code(text)

###########################################################################
#                           has_classes                                   #
###########################################################################

def has_classes(code: str) -> bool:
    """Checks if the given Python code contains no class definitions.

    Args:
        code: A string containing Python code to be analyzed.

    Returns:
        bool: True if the code contains no class definitions, False otherwise.

    Raises:
        SyntaxError: If the provided code is not valid Python syntax.
    """
    try:
        syntax_tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax: {e}") from e

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ClassDef):
            return True
    return False

###########################################################################
#                           has_invalid_chars                            #
###########################################################################

def has_invalid_chars(text):
    """
    判断给定的代码文本是否包含乱码或非英语字符。

    :param text: 要检查的代码文件内容
    :return: 如果包含乱码或非英语字符，返回 True;否则返回 False
    """
    # 设定 ASCII 范围：32-126为常见可打印字符，常用于代码和文本中
    allowed_ascii_ranges = [(0, 126)]

    # 遍历每个字符的 ASCII 值
    for char in text:
        ascii_value = ord(char)
        # 检查字符是否在允许的 ASCII 范围内
        if not any(start <= ascii_value <= end for start, end in allowed_ascii_ranges):
            return True

    # 如果所有字符都在允许的范围内，返回 False
    return False

###########################################################################
#                           has_disallowed_import                         #
###########################################################################

def extract_top_level_imports(source_code: str) -> Set[str]:
    """Extracts all top-level import module names from the given Python code.

    Args:
        source_code: A string containing Python code.

    Returns:
        A set of strings, each representing the top-level module name of an import.

    Example:
        >>> code = "import os\\nfrom sys import path\\nimport numpy as np"
        >>> extract_top_level_imports(code)
        {'os', 'sys', 'numpy'}
    """
    syntax_tree = ast.parse(source_code)
    top_level_modules = set()

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Only keep the first part of the module name
                if alias.name:
                    module_name = alias.name.split('.')[0]
                    top_level_modules.add(module_name)
        elif isinstance(node, ast.ImportFrom):
                # print(node.level)
                # Only keep the first part of the module name
                if node.module:
                    module_name = node.module.split('.')[0]
                    top_level_modules.add(module_name)

    return top_level_modules

def has_disallowed_imports(source_code: str, allowed_imports: Set[str]) -> bool:
    """Checks if the code contains any imports that are not in the set of allowed imports.

    Args:
        source_code: A string containing Python code to be analyzed.
        allowed_imports: A set of strings representing allowed import names.

    Returns:
        bool: True if there are any disallowed imports, False if all imports are allowed.

    Raises:
        ValueError: If the provided allowed_imports set is empty.
    """
    if not allowed_imports:
        raise ValueError("The set of allowed imports cannot be empty.")

    actual_imports = extract_top_level_imports(source_code)
    return not actual_imports.issubset(allowed_imports)


###########################################################################
#                           has_io_operations                             #
###########################################################################

def _is_io_operation(node: ast.AST) -> bool:
    """Helper function to determine if a node represents an I/O operation.

    Args:
        node: An AST node to be checked.

    Returns:
        bool: True if the node represents an I/O operation, False otherwise.
    """
    if isinstance(node, ast.Call):
        return (_is_direct_io_call(node) or
                _is_file_method_call(node) or
                _is_os_io_operation(node))
    elif isinstance(node, ast.With):
        return _is_file_context_manager(node)
    return False


def _is_direct_io_call(node: ast.Call) -> bool:
    return (isinstance(node.func, ast.Name) and
            node.func.id in ('open', 'input'))


def _is_file_method_call(node: ast.Call) -> bool:
    return (isinstance(node.func, ast.Attribute) and
            node.func.attr in ('read', 'write', 'readline', 'readlines',
                               'writelines', 'writeline', 'seek', 'tell'))


def _is_file_context_manager(node: ast.With) -> bool:
    for item in node.items:
        if isinstance(item.context_expr, ast.Call):
            func = item.context_expr.func
            if ((isinstance(func, ast.Name) and func.id == 'open') or
                (isinstance(func, ast.Attribute) and func.attr == 'open')):
                return True
    return False


def _is_os_io_operation(node: ast.Call) -> bool:
    return (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'os' and
            node.func.attr in ('remove', 'rename', 'mkdir', 'rmdir',
                               'makedirs', 'removedirs', 'chmod', 'chown'))


def has_io_operations(code: str) -> bool:
    """Checks if the given Python code contains input/output operations.

    This function analyzes the abstract syntax tree of the provided code to detect
    various forms of I/O operations, including file operations and user input.

    Args:
        code: A string containing Python code to be analyzed.

    Returns:
        bool: True if I/O operations are detected, False otherwise.

    Raises:
        SyntaxError: If the provided code is not valid Python syntax.
    """
    syntax_tree = ast.parse(code)

    for node in ast.walk(syntax_tree):
        if _is_io_operation(node):
            return True

    return False

###########################################################################
#                           has_io_operations                             #
###########################################################################

def count_top_level_functions(code: str) -> int:
    """
    Count the number of top-level functions in the given Python code string.
    
    Args:
        code: A string containing Python code.
    
    Returns:
        An integer representing the number of top-level functions in the code.
    """
    tree = ast.parse(code)

    function_count = sum(1 for node in tree.body if isinstance(node, ast.FunctionDef))
    return function_count

###########################################################################
#                           have_args_in_function                         #
###########################################################################

def have_args_in_function(code: str) -> bool:
    """
    Check if the top-level function in the given Python code has arguments.

    Args:
        code: A string containing Python code.

    Returns:
        True if the function has arguments, False otherwise.
    """
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not bool(node.args.args):
                return False
    return True

###########################################################################
#                           have_return_in_function                         #
###########################################################################

def have_return_in_function(code: str) -> bool:
    """
    Check if the top-level function in the given Python code contains at least one return statement.

    Args:
        code: A string containing Python code.

    Returns:
        True if there is at least one return statement, False otherwise.
    """
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not any(isinstance(child, ast.Return) for child in node.body):
                return False
    return True

###########################################################################
#                           has_io_operations                             #
###########################################################################

def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps

def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]

    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)

    return visited

def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None

def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))

def calculate_dependency_depth(function_deps: Dict[str, Set[str]]) -> Dict[str, int]:
    depth = {}
    visited = set()

    def dfs(func):
        if func in visited:
            return depth.get(func, 0)
        visited.add(func)
        if func not in function_deps:
            depth[func] = 0
            return 0
        max_depth = 0
        for dep in function_deps[func]:
            max_depth = max(max_depth, dfs(dep) + 1)
        depth[func] = max_depth
        return max_depth

    for func in function_deps:
        if func not in visited:
            dfs(func)

    return depth

def extract_entry_point(text: str) -> List[Tuple[str, int, Set[str]]]:
    
    try:
        tree = ast.parse(text)
        
        function_defs = [(node.name, node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        function_deps = get_deps(function_defs)
        
        # Filter out dependencies that are not defined functions
        defined_functions = set(func[0] for func in function_defs)
        for func, deps in function_deps.items():
            function_deps[func] = deps.intersection(defined_functions)
        
        depth = calculate_dependency_depth(function_deps)
        
        # Sort functions by depth (deepest first), then by name
        sorted_functions = sorted(depth.items(), key=lambda x: (-x[1], x[0]))
        
        # Find the first function that is not 'main'
        for func, d in sorted_functions:
            if func != 'main' and func != '__init__' and func != 'test' and func != 'check' and func != 'Solution' and func != 'solution':
                return func
        
        # If no function other than 'main' is found, return None
        return None
    except Exception as e:
        return None

###########################################################################
#                           extract_functions                             #
###########################################################################

import ast
from typing import List, Dict, Any, Optional

def extract_functions(code: str) -> List[Dict[str, Any]]:
    """Extract all function bodies from the given Python code, including detailed parameter and return type information.

    Args:
        code: A string containing Python code.

    Returns:
        A list of dictionaries, each containing information about a function:
        - 'name': The name of the function.
        - 'body': The complete function body as a string.
        - 'lineno': The starting line number of the function.
        - 'end_lineno': The ending line number of the function.
        - 'params': A list of dictionaries, each containing:
            - 'name': The name of the parameter.
            - 'type': The annotated type of the parameter, if any.
        - 'return_type': The annotated return type, if any.

    Raises:
        SyntaxError: If the input code is not valid Python syntax.
    """
    functions = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in the provided code: {e}")
        return functions

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_body = ast.unparse(node)
            
            # Extract parameter information
            params = []
            for arg in node.args.args:
                param_info = {
                    'name': arg.arg,
                    'type': ast.unparse(arg.annotation) if arg.annotation else None
                }
                params.append(param_info)
            
            # Extract return type
            return_type = ast.unparse(node.returns) if node.returns else None
            
            functions.append({
                'name': node.name,
                'body': function_body,
                'lineno': node.lineno,
                'end_lineno': node.end_lineno,
                'params': params,
                'return_type': return_type
            })

    return functions

###########################################################################
#                           stream_jsonl                                  #
###########################################################################

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

###########################################################################
#                           write_jsonl                                #
###########################################################################

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def refine_text(text: str) -> str:
    text =  text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"

def extract_content(text: str, start: str = "", end: str = "") -> str:
    # 如果开始和结束标识都为空，返回整个字符串
    if start == "" and end == "":
        return text
    
    # 如果开始为空，返回从开头到结束标识符的内容
    if start == "":
        end_index = text.find(end)
        return text[:end_index] if end_index != -1 else text
    
    # 如果结束为空，返回从开始标识符到字符串的末尾
    if end == "":
        start_index = text.find(start)
        return text[start_index + len(start):] if start_index != -1 else ""

    # 查找开始和结束标识符的位置
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))

    # 检查标识符是否存在于字符串中
    if start_index == -1 or end_index == -1:
        return ""

    # 提取并返回开始和结束标识符之间的内容
    return text[start_index + len(start):end_index]

python_pattern = r"```python[ \t]*[\r\n]+(.*?)[ \t]*[\r\n]+```"
python_re = re.compile(python_pattern, re.DOTALL | re.IGNORECASE)

def python_extract(text: str) -> str:
    match = python_re.search(text)
    if match:
        return match.group(1)
    else:
        return ""