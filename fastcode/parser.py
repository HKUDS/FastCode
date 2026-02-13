"""
Code Parser - AST-based code parsing for multiple languages
"""

import ast
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import libcst as cst

from .utils import (
    get_language_from_extension,
    get_file_extension,
    calculate_code_complexity,
    clean_docstring,
)


@dataclass
class FunctionInfo:
    """Function/method information"""
    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    is_async: bool
    is_method: bool
    class_name: Optional[str]
    decorators: List[str]
    complexity: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassInfo:
    """Class information"""
    name: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    bases: List[str]
    methods: List[FunctionInfo]  # <--- CHANGED: List[str] to List[FunctionInfo]
    decorators: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportInfo:
    """Import statement information"""
    module: str
    names: List[str]
    is_from: bool
    line: int
    level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileParseResult:
    """Result of parsing a file"""
    file_path: str
    language: str
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    imports: List[ImportInfo]
    module_docstring: Optional[str]
    total_lines: int
    code_lines: int
    comment_lines: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "imports": [i.to_dict() for i in self.imports],
            "module_docstring": self.module_docstring,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
        }


class CodeParser:
    """Parse code files and extract structured information"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parser_config = config.get("parser", {})
        self.logger = logging.getLogger(__name__)
        
        self.extract_docstrings = self.parser_config.get("extract_docstrings", True)
        self.extract_comments = self.parser_config.get("extract_comments", True)
        self.extract_imports = self.parser_config.get("extract_imports", True)
        self.compute_complexity = self.parser_config.get("compute_complexity", True)
        self.max_function_lines = self.parser_config.get("max_function_lines", 1000)
        
        # self._skipped_node_log_count = 0
    
    def parse_file(self, file_path: str, content: str) -> Optional[FileParseResult]:
        """
        Parse a code file

        Args:
            file_path: Path to file
            content: File content

        Returns:
            FileParseResult or None if parsing failed
        """
        ext = get_file_extension(file_path)
        language = get_language_from_extension(ext)

        # Route to appropriate parser
        if language == "python":
            return self._parse_python(file_path, content)
        elif language == "javascript":
            return self._parse_javascript(file_path, content, language)
        elif language == "typescript":
            return self._parse_typescript(file_path, content, language)
        elif language in ["c", "cpp"]:
            return self._parse_c_cpp(file_path, content, language)
        elif language == "rust":
            return self._parse_rust(file_path, content)
        elif language == "csharp":
            return self._parse_csharp(file_path, content)
        else:
            # For unsupported languages, return basic info
            return self._parse_generic(file_path, content, language)
    
    def _fix_common_syntax_errors(self, content: str) -> str:
        """
        Fix common syntax errors in generated code.
        
        Some generated files may have syntax errors like:
        - except Exception as exc as exc: (duplicate 'as' clause)
        
        Args:
            content: File content that may contain syntax errors
            
        Returns:
            Content with common syntax errors fixed
        """
        import re
        
        # Fix duplicate 'as' clause in except statements
        # Pattern: except SomeException as var as var:
        content = re.sub(
            r'except\s+(\w+)\s+as\s+(\w+)\s+as\s+\2\s*:',
            r'except \1 as \2:',
            content
        )
        
        return content
    
    def _strip_markdown_code_fences(self, content: str) -> str:
        """
        Strip markdown code fences from file content.
        
        Some generated files may have ```python at the start and ``` at the end.
        This method removes these markers to allow proper parsing.
        
        Args:
            content: File content that may contain markdown fences
            
        Returns:
            Content with markdown fences removed
        """
        lines = content.split('\n')
        
        # Check if first line is a markdown code fence (e.g., ```python, ```javascript)
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        
        # Remove trailing lines that are markdown fences or empty
        # Work backwards from the end to handle cases where ``` is not the last line
        while lines:
            last_line = lines[-1].strip()
            if last_line == '```' or last_line == '':
                lines = lines[:-1]
            else:
                break
        
        return '\n'.join(lines)

    
    def _parse_python(self, file_path: str, content: str) -> Optional[FileParseResult]:
        """Parse Python file using AST"""
        # Fix common syntax errors in generated code
        content = self._fix_common_syntax_errors(content)
        
        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return None
        
        classes = []
        functions = []
        imports = []
        module_docstring = ast.get_docstring(tree)
        
        # Extract imports
        if self.extract_imports:
            imports = self._extract_python_imports(tree)
        
        # Extract classes and functions
        # for node in tree.body:
        #     if isinstance(node, ast.ClassDef):
        #         class_info = self._extract_python_class(node)
        #         if class_info:
        #             classes.append(class_info)
            
        #     elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        #         func_info = self._extract_python_function(node)
        #         if func_info:
        #             functions.append(func_info)
        
        # # fastest version 
        # for node in tree.body:
        #     if isinstance(node, ast.ClassDef):
        #         class_info = self._extract_python_class(node)
        #         if class_info:
        #             classes.append(class_info)
            
        #     elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        #         func_info = self._extract_python_function(node)
        #         if func_info:
        #             functions.append(func_info)
        #     else:
        #         node_type = type(node).__name__
        #         if self._skipped_node_log_count < 100:
        #             if isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)):
        #                 hidden_defs = []
        #                 for child in ast.walk(node):
        #                     if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        #                         name = getattr(child, 'name', 'unknown')
        #                         hidden_defs.append(f"{type(child).__name__}:{name}")
                        
        #                 if hidden_defs:
        #                     self.logger.warning(
        #                         f"[EDGE LOSS INVESTIGATION] In {file_path}: Skipped top-level node '{type(node).__name__}' (Line {node.lineno}). "
        #                         f"It contains {len(hidden_defs)} definitions that Code B is ignoring: {hidden_defs}"
        #                     )
        #                     self._skipped_node_log_count += 1
                            
        #                     if self._skipped_node_log_count == 100:
        #                         self.logger.warning("[EDGE LOSS INVESTIGATION] Limit reached (100 logs). Further skip warnings will be suppressed.")

        def _visit_nodes(node_list, parent_scope=None):
            """
            Recursively visit nodes, drilling into If/Try/Else blocks,
            but capturing Function/Class definitions.
            """
            for node in node_list:
                # 1. Capture Class Definitions
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_python_class(node)
                    if class_info:
                        classes.append(class_info)
                    # We do NOT drill into classes here because _extract_python_class 
                    # already handles its internal methods.
                
                # 2. Capture Function Definitions
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Option: Pass 'parent_scope' if you want to track nesting
                    func_info = self._extract_python_function(node)
                    if func_info:
                        functions.append(func_info)
                    # We do NOT drill into functions (avoiding nested functions)
                    # UNLESS you specifically want to support them like Code A did.
                    # Current logic: Top-level definitions only (including those in If/Try).
                
                # 3. Smart Drill-down (The Fix for compatibility.py)
                elif isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)):
                    # Drill down into the body of these blocks
                    _visit_nodes(node.body, parent_scope)
                    
                    # Handle 'else' blocks for If/Try/For/While
                    if hasattr(node, 'orelse') and node.orelse:
                        _visit_nodes(node.orelse, parent_scope)
                    
                    # Handle 'finalbody' for Try
                    if hasattr(node, 'finalbody') and node.finalbody:
                        _visit_nodes(node.finalbody, parent_scope)

        # Start the visit from the root body
        _visit_nodes(tree.body)
        # ------------------------------------------

        # Count lines
        lines = content.split("\n")
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        
        return FileParseResult(
            file_path=file_path,
            language="python",
            classes=classes,
            functions=functions,
            imports=imports,
            module_docstring=clean_docstring(module_docstring) if module_docstring else None,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
        )
    
    def _extract_python_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract import statements from Python AST"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.asname if alias.asname else alias.name],
                        is_from=False,
                        level=0,  # ast.Import always uses absolute imports
                        line=node.lineno,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from=True,
                    level=node.level,  # Key fix: capture relative import level
                    line=node.lineno,
                ))

        return imports
    
    def _extract_python_class(self, node: ast.ClassDef) -> Optional[ClassInfo]:
        """Extract class information from Python AST node"""
        try:
            docstring = ast.get_docstring(node)
            
            # Extract base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
            
            # Extract methods (FULL INFO)
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.logger.debug(f"[DEBUG PARSER] Found method '{item.name}' in class '{node.name}'")
                    # Use the existing function extractor, passing the class name context
                    func_info = self._extract_python_function(item, class_name=node.name)
                    if func_info:
                        methods.append(func_info)
                    else:
                        self.logger.debug(f"[DEBUG PARSER] Found method '{item.name}' in class '{node.name}'")
            
            self.logger.debug((f"[DEBUG PARSER] Class '{node.name}' extracted with {len(methods)} methods"))
            if methods:
                self.logger.debug(f"[DEBUG PARSER] Method type: {type(methods[0])} (Should be FunctionInfo)")
            
            # Extract decorators
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
            
            return ClassInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                docstring=clean_docstring(docstring) if docstring else None,
                bases=bases,
                methods=methods,
                decorators=decorators,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract class info: {e}")
            return None
    
    def _extract_python_function(self, node: ast.FunctionDef, 
                                  class_name: Optional[str] = None) -> Optional[FunctionInfo]:
        """Extract function information from Python AST node"""
        try:
            # Skip if too long
            if node.end_lineno and (node.end_lineno - node.lineno) > self.max_function_lines:
                self.logger.debug(f"Skipping long function: {node.name}")
                return None
            
            docstring = ast.get_docstring(node)
            
            # Extract parameters
            parameters = []
            for arg in node.args.args:
                param_name = arg.arg
                if arg.annotation:
                    # Try to get type annotation
                    param_name += f": {ast.unparse(arg.annotation)}"
                parameters.append(param_name)
            
            # Extract return type
            return_type = None
            if node.returns:
                try:
                    return_type = ast.unparse(node.returns)
                except Exception:
                    pass
            
            # Extract decorators
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
            
            # Calculate complexity
            complexity = 1
            if self.compute_complexity:
                complexity = self._calculate_python_complexity(node)
            
            return FunctionInfo(
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                docstring=clean_docstring(docstring) if docstring else None,
                parameters=parameters,
                return_type=return_type,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_method=class_name is not None,
                class_name=class_name,
                decorators=decorators,
                complexity=complexity,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract function info: {e}")
            return None
    
    def _calculate_python_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for Python function"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _parse_javascript(self, file_path: str, content: str, language: str) -> Optional[FileParseResult]:
        """Parse JavaScript/TypeScript file using tree-sitter"""
        from .tree_sitter_parser import TSParser
        from tree_sitter import Query

        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)

        try:
            # Initialize parser for JavaScript
            ts_parser = TSParser(language='javascript')
            tree = ts_parser.parse(content)
            if not tree:
                return self._parse_generic(file_path, content, language)

            root_node = tree.root_node

            classes = []
            functions = []
            imports = []
            module_docstring = None

            # Extract module-level comment as docstring
            if self.extract_docstrings:
                module_docstring = self._extract_js_module_docstring(content, root_node)

            # Extract imports
            if self.extract_imports:
                imports = self._extract_js_imports(root_node, content)

            # Extract classes and functions
            self._extract_js_classes_and_functions(
                root_node, content, classes, functions
            )

            # Count lines
            lines = content.split("\n")
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("//", "/*", "*")))
            comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*")))

            return FileParseResult(
                file_path=file_path,
                language=language,
                classes=classes,
                functions=functions,
                imports=imports,
                module_docstring=clean_docstring(module_docstring) if module_docstring else None,
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            return self._parse_generic(file_path, content, language)

    def _extract_js_module_docstring(self, content: str, root_node) -> Optional[str]:
        """Extract module-level documentation from JavaScript file"""
        # Look for leading comment blocks
        code_bytes = content.encode('utf-8')
        for child in root_node.children:
            if child.type == 'comment':
                comment_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                # Clean up comment markers
                if comment_text.startswith('//'):
                    return comment_text[2:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
                return comment_text
        return None

    def _extract_js_imports(self, root_node, content: str) -> List[ImportInfo]:
        """Extract import statements from JavaScript AST"""
        imports = []
        code_bytes = content.encode('utf-8')

        def visit_node(node):
            if node.type == 'import_statement':
                # Extract import information
                module_node = None
                names = []

                for child in node.children:
                    if child.type == 'string':
                        module_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        module_node = module_text.strip('"\'')
                    elif child.type == 'import_clause':
                        # Extract imported names
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                names.append(code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8'))
                            elif subchild.type == 'named_imports':
                                for spec in subchild.children:
                                    if spec.type == 'import_specifier':
                                        for id_node in spec.children:
                                            if id_node.type == 'identifier':
                                                names.append(code_bytes[id_node.start_byte:id_node.end_byte].decode('utf-8'))

                if module_node:
                    imports.append(ImportInfo(
                        module=module_node,
                        names=names if names else ['*'],
                        is_from=True,
                        line=node.start_point[0] + 1,
                        level=0
                    ))

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return imports

    def _extract_js_classes_and_functions(self, root_node, content: str, classes: List, functions: List):
        """Extract classes and functions from JavaScript AST"""
        code_bytes = content.encode('utf-8')

        def visit_node(node, current_class=None):
            if node.type == 'class_declaration':
                class_info = self._extract_js_class(node, content, code_bytes)
                if class_info:
                    classes.append(class_info)
            elif node.type in ('function_declaration', 'arrow_function', 'function'):
                # Only extract top-level functions or methods
                func_info = self._extract_js_function(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            elif node.type == 'method_definition':
                # This is a class method
                func_info = self._extract_js_method(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            else:
                # Recursively visit children
                for child in node.children:
                    if node.type == 'class_declaration':
                        visit_node(child, node)
                    else:
                        visit_node(child, current_class)

        visit_node(root_node)

    def _extract_js_class(self, node, content: str, code_bytes: bytes) -> Optional[ClassInfo]:
        """Extract class information from JavaScript AST node"""
        try:
            # Get class name
            name_node = None
            for child in node.children:
                if child.type == 'identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            class_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract base class (extends)
            bases = []
            for child in node.children:
                if child.type == 'class_heritage':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            bases.append(code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8'))

            # Extract methods
            methods = []
            for child in node.children:
                if child.type == 'class_body':
                    for method_node in child.children:
                        if method_node.type == 'method_definition':
                            method_info = self._extract_js_method(method_node, content, code_bytes, class_name)
                            if method_info:
                                methods.append(method_info)

            # Extract docstring (JSDoc comment before class)
            docstring = self._extract_js_docstring(node, content, code_bytes)

            return ClassInfo(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
                methods=methods,
                decorators=[]
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract JS class: {e}")
            return None

    def _extract_js_function(self, node, content: str, code_bytes: bytes, class_name: Optional[str] = None) -> Optional[FunctionInfo]:
        """Extract function information from JavaScript AST node"""
        try:
            # Get function name
            name_node = None
            for child in node.children:
                if child.type == 'identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            func_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract parameters
            parameters = []
            for child in node.children:
                if child.type == 'formal_parameters':
                    for param_node in child.children:
                        if param_node.type in ('identifier', 'required_parameter', 'optional_parameter'):
                            param_text = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf-8')
                            parameters.append(param_text)

            # Extract docstring
            docstring = self._extract_js_docstring(node, content, code_bytes)

            # Check if async
            is_async = any(child.type == 'async' or child.text == b'async' for child in node.children)

            return FunctionInfo(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                parameters=parameters,
                return_type=None,
                is_async=is_async,
                is_method=class_name is not None,
                class_name=class_name,
                decorators=[],
                complexity=1
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract JS function: {e}")
            return None

    def _extract_js_method(self, node, content: str, code_bytes: bytes, class_name: Optional[str]) -> Optional[FunctionInfo]:
        """Extract method information from JavaScript class"""
        try:
            # Get method name
            name_node = None
            for child in node.children:
                if child.type == 'property_identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            method_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract parameters
            parameters = []
            for child in node.children:
                if child.type == 'formal_parameters':
                    for param_node in child.children:
                        if param_node.type in ('identifier', 'required_parameter', 'optional_parameter'):
                            param_text = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf-8')
                            parameters.append(param_text)

            # Extract docstring
            docstring = self._extract_js_docstring(node, content, code_bytes)

            # Check if async
            is_async = any(child.type == 'async' or child.text == b'async' for child in node.children)

            return FunctionInfo(
                name=method_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                parameters=parameters,
                return_type=None,
                is_async=is_async,
                is_method=True,
                class_name=class_name,
                decorators=[],
                complexity=1
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract JS method: {e}")
            return None

    def _extract_js_docstring(self, node, content: str, code_bytes: bytes) -> Optional[str]:
        """Extract JSDoc comment before a node"""
        # Look for comment node immediately before this node
        parent = node.parent
        if parent:
            prev_sibling = None
            for i, child in enumerate(parent.children):
                if child == node and i > 0:
                    prev_sibling = parent.children[i - 1]
                    break

            if prev_sibling and prev_sibling.type == 'comment':
                comment_text = code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf-8')
                # Clean up JSDoc comment markers
                if comment_text.startswith('/**') and comment_text.endswith('*/'):
                    return comment_text[3:-2].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
                elif comment_text.startswith('//'):
                    return comment_text[2:].strip()

        return None

    def _parse_typescript(self, file_path: str, content: str, language: str) -> Optional[FileParseResult]:
        """Parse TypeScript file using tree-sitter"""
        from .tree_sitter_parser import TSParser

        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)

        try:
            # Initialize parser for TypeScript
            lang = 'tsx' if language == 'tsx' or file_path.endswith('.tsx') else 'typescript'
            ts_parser = TSParser(language=lang)
            tree = ts_parser.parse(content)
            if not tree:
                return self._parse_generic(file_path, content, language)

            root_node = tree.root_node

            classes = []
            functions = []
            imports = []
            module_docstring = None

            # Extract module-level comment as docstring
            if self.extract_docstrings:
                module_docstring = self._extract_js_module_docstring(content, root_node)

            # Extract imports (same as JS)
            if self.extract_imports:
                imports = self._extract_js_imports(root_node, content)

            # Extract classes, functions, and TypeScript-specific constructs
            self._extract_ts_classes_and_functions(
                root_node, content, classes, functions
            )

            # Count lines
            lines = content.split("\n")
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("//", "/*", "*")))
            comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*")))

            return FileParseResult(
                file_path=file_path,
                language=language,
                classes=classes,
                functions=functions,
                imports=imports,
                module_docstring=clean_docstring(module_docstring) if module_docstring else None,
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            return self._parse_generic(file_path, content, language)

    def _extract_ts_classes_and_functions(self, root_node, content: str, classes: List, functions: List):
        """Extract classes, interfaces, and functions from TypeScript AST"""
        code_bytes = content.encode('utf-8')

        def visit_node(node, current_class=None):
            # TypeScript classes
            if node.type in ('class_declaration', 'interface_declaration'):
                class_info = self._extract_ts_class(node, content, code_bytes)
                if class_info:
                    classes.append(class_info)
            # TypeScript functions
            elif node.type in ('function_declaration', 'arrow_function', 'function'):
                func_info = self._extract_js_function(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            elif node.type in ('method_definition', 'method_signature'):
                func_info = self._extract_js_method(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            else:
                # Recursively visit children
                for child in node.children:
                    if node.type in ('class_declaration', 'interface_declaration'):
                        visit_node(child, node)
                    else:
                        visit_node(child, current_class)

        visit_node(root_node)

    def _extract_ts_class(self, node, content: str, code_bytes: bytes) -> Optional[ClassInfo]:
        """Extract class/interface information from TypeScript AST node"""
        try:
            # Get class/interface name
            name_node = None
            for child in node.children:
                if child.type == 'type_identifier' or child.type == 'identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            class_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract base classes/interfaces (extends/implements)
            bases = []
            for child in node.children:
                if child.type in ('class_heritage', 'extends_clause', 'implements_clause'):
                    for subchild in child.children:
                        if subchild.type in ('identifier', 'type_identifier'):
                            bases.append(code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8'))

            # Extract methods
            methods = []
            for child in node.children:
                if child.type in ('class_body', 'interface_body', 'object_type'):
                    for method_node in child.children:
                        if method_node.type in ('method_definition', 'method_signature'):
                            method_info = self._extract_js_method(method_node, content, code_bytes, class_name)
                            if method_info:
                                methods.append(method_info)

            # Extract docstring (TSDoc comment before class)
            docstring = self._extract_js_docstring(node, content, code_bytes)

            return ClassInfo(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
                methods=methods,
                decorators=[]
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract TS class/interface: {e}")
            return None

    def _parse_c_cpp(self, file_path: str, content: str, language: str) -> Optional[FileParseResult]:
        """Parse C/C++ file using tree-sitter"""
        from .tree_sitter_parser import TSParser

        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)

        try:
            # Initialize parser for C or C++
            lang = 'cpp' if language == 'cpp' else 'c'
            ts_parser = TSParser(language=lang)
            tree = ts_parser.parse(content)
            if not tree:
                return self._parse_generic(file_path, content, language)

            root_node = tree.root_node

            classes = []
            functions = []
            imports = []
            module_docstring = None

            # Extract module-level comment as docstring
            if self.extract_docstrings:
                module_docstring = self._extract_c_module_docstring(content, root_node)

            # Extract includes
            if self.extract_imports:
                imports = self._extract_c_includes(root_node, content)

            # Extract classes/structs and functions
            self._extract_c_classes_and_functions(
                root_node, content, classes, functions, language
            )

            # Count lines
            lines = content.split("\n")
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("//", "/*", "*")))
            comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*")))

            return FileParseResult(
                file_path=file_path,
                language=language,
                classes=classes,
                functions=functions,
                imports=imports,
                module_docstring=clean_docstring(module_docstring) if module_docstring else None,
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            return self._parse_generic(file_path, content, language)

    def _extract_c_module_docstring(self, content: str, root_node) -> Optional[str]:
        """Extract module-level documentation from C/C++ file"""
        code_bytes = content.encode('utf-8')
        for child in root_node.children:
            if child.type == 'comment':
                comment_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                # Clean up comment markers
                if comment_text.startswith('//'):
                    return comment_text[2:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
                return comment_text
        return None

    def _extract_c_includes(self, root_node, content: str) -> List[ImportInfo]:
        """Extract #include statements from C/C++ AST"""
        imports = []
        code_bytes = content.encode('utf-8')

        def visit_node(node):
            if node.type == 'preproc_include':
                # Extract include path
                for child in node.children:
                    if child.type in ('string_literal', 'system_lib_string'):
                        include_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                        include_text = include_text.strip('<>"')
                        imports.append(ImportInfo(
                            module=include_text,
                            names=['*'],
                            is_from=False,
                            line=node.start_point[0] + 1,
                            level=0
                        ))

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return imports

    def _extract_c_classes_and_functions(self, root_node, content: str, classes: List, functions: List, language: str):
        """Extract classes/structs and functions from C/C++ AST"""
        code_bytes = content.encode('utf-8')

        def visit_node(node, current_class=None):
            # C++ classes or C structs
            if node.type in ('class_specifier', 'struct_specifier'):
                class_info = self._extract_c_class(node, content, code_bytes, language)
                if class_info:
                    classes.append(class_info)
            # Functions
            elif node.type == 'function_definition':
                func_info = self._extract_c_function(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            else:
                # Recursively visit children
                for child in node.children:
                    if node.type in ('class_specifier', 'struct_specifier'):
                        visit_node(child, node)
                    else:
                        visit_node(child, current_class)

        visit_node(root_node)

    def _extract_c_class(self, node, content: str, code_bytes: bytes, language: str) -> Optional[ClassInfo]:
        """Extract class/struct information from C/C++ AST node"""
        try:
            # Get class/struct name
            name_node = None
            for child in node.children:
                if child.type in ('type_identifier', 'identifier'):
                    name_node = child
                    break

            if not name_node:
                return None

            class_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract base classes (C++ only)
            bases = []
            if language == 'cpp':
                for child in node.children:
                    if child.type == 'base_class_clause':
                        for subchild in child.children:
                            if subchild.type in ('type_identifier', 'identifier'):
                                bases.append(code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8'))

            # Extract methods (functions within class/struct)
            methods = []
            for child in node.children:
                if child.type == 'field_declaration_list':
                    for member_node in child.children:
                        if member_node.type == 'function_definition':
                            method_info = self._extract_c_function(member_node, content, code_bytes, class_name)
                            if method_info:
                                methods.append(method_info)

            # Extract docstring (Doxygen comment before class)
            docstring = self._extract_c_docstring(node, content, code_bytes)

            return ClassInfo(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
                methods=methods,
                decorators=[]
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract C/C++ class: {e}")
            return None

    def _extract_c_function(self, node, content: str, code_bytes: bytes, class_name: Optional[str] = None) -> Optional[FunctionInfo]:
        """Extract function information from C/C++ AST node"""
        try:
            # Get function name from declarator
            func_name = None
            parameters = []
            return_type = None

            # Navigate to function declarator
            declarator_node = None
            for child in node.children:
                if child.type == 'function_declarator':
                    declarator_node = child
                    break

            if not declarator_node:
                return None

            # Get function name
            for child in declarator_node.children:
                if child.type in ('identifier', 'field_identifier'):
                    func_name = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                elif child.type == 'parameter_list':
                    # Extract parameters
                    for param_node in child.children:
                        if param_node.type == 'parameter_declaration':
                            param_text = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf-8')
                            parameters.append(param_text)

            if not func_name:
                return None

            # Extract return type (primitive or type identifier before declarator)
            for child in node.children:
                if child.type in ('primitive_type', 'type_identifier'):
                    return_type = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                    break

            # Extract docstring
            docstring = self._extract_c_docstring(node, content, code_bytes)

            return FunctionInfo(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                is_async=False,
                is_method=class_name is not None,
                class_name=class_name,
                decorators=[],
                complexity=1
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract C/C++ function: {e}")
            return None

    def _extract_c_docstring(self, node, content: str, code_bytes: bytes) -> Optional[str]:
        """Extract Doxygen/comment before a C/C++ node"""
        parent = node.parent
        if parent:
            prev_sibling = None
            for i, child in enumerate(parent.children):
                if child == node and i > 0:
                    prev_sibling = parent.children[i - 1]
                    break

            if prev_sibling and prev_sibling.type == 'comment':
                comment_text = code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf-8')
                # Clean up Doxygen/regular comment markers
                if comment_text.startswith('/**') and comment_text.endswith('*/'):
                    return comment_text[3:-2].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
                elif comment_text.startswith('///'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('//'):
                    return comment_text[2:].strip()

        return None

    def _parse_rust(self, file_path: str, content: str) -> Optional[FileParseResult]:
        """Parse Rust file using tree-sitter"""
        from .tree_sitter_parser import TSParser

        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)

        try:
            # Initialize parser for Rust
            ts_parser = TSParser(language='rust')
            tree = ts_parser.parse(content)
            if not tree:
                return self._parse_generic(file_path, content, 'rust')

            root_node = tree.root_node

            classes = []  # Will include structs, traits, impls
            functions = []
            imports = []
            module_docstring = None

            # Extract module-level documentation
            if self.extract_docstrings:
                module_docstring = self._extract_rust_module_docstring(content, root_node)

            # Extract use statements
            if self.extract_imports:
                imports = self._extract_rust_imports(root_node, content)

            # Extract structs, traits, impls, and functions
            self._extract_rust_items(
                root_node, content, classes, functions
            )

            # Count lines
            lines = content.split("\n")
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("//", "/*", "*")))
            comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*")))

            return FileParseResult(
                file_path=file_path,
                language='rust',
                classes=classes,
                functions=functions,
                imports=imports,
                module_docstring=clean_docstring(module_docstring) if module_docstring else None,
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            return self._parse_generic(file_path, content, 'rust')

    def _extract_rust_module_docstring(self, content: str, root_node) -> Optional[str]:
        """Extract module-level documentation from Rust file"""
        code_bytes = content.encode('utf-8')
        for child in root_node.children:
            if child.type in ('line_comment', 'block_comment'):
                comment_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                # Rust doc comments start with /// or //!
                if comment_text.startswith('//!'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('///'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
        return None

    def _extract_rust_imports(self, root_node, content: str) -> List[ImportInfo]:
        """Extract 'use' statements from Rust AST"""
        imports = []
        code_bytes = content.encode('utf-8')

        def visit_node(node):
            if node.type == 'use_declaration':
                # Extract use path
                use_text = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                # Simple extraction: get the full use statement
                use_text = use_text.replace('use ', '').replace(';', '').strip()
                imports.append(ImportInfo(
                    module=use_text,
                    names=['*'],
                    is_from=True,
                    line=node.start_point[0] + 1,
                    level=0
                ))

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return imports

    def _extract_rust_items(self, root_node, content: str, classes: List, functions: List):
        """Extract structs, traits, impls, and functions from Rust AST"""
        code_bytes = content.encode('utf-8')

        def visit_node(node, current_class=None):
            # Rust structs, traits, impls
            if node.type in ('struct_item', 'trait_item', 'impl_item'):
                class_info = self._extract_rust_type(node, content, code_bytes)
                if class_info:
                    classes.append(class_info)
            # Rust functions
            elif node.type == 'function_item':
                func_info = self._extract_rust_function(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            else:
                # Recursively visit children
                for child in node.children:
                    if node.type in ('struct_item', 'trait_item', 'impl_item'):
                        visit_node(child, node)
                    else:
                        visit_node(child, current_class)

        visit_node(root_node)

    def _extract_rust_type(self, node, content: str, code_bytes: bytes) -> Optional[ClassInfo]:
        """Extract struct/trait/impl information from Rust AST node"""
        try:
            # Get type name
            name_node = None
            for child in node.children:
                if child.type == 'type_identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            type_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract trait bounds or impl target
            bases = []
            if node.type == 'impl_item':
                # For impl blocks, extract the trait being implemented
                for child in node.children:
                    if child.type == 'type_identifier' and child != name_node:
                        bases.append(code_bytes[child.start_byte:child.end_byte].decode('utf-8'))

            # Extract methods (functions within struct/trait/impl)
            methods = []
            for child in node.children:
                if child.type == 'declaration_list':
                    for item_node in child.children:
                        if item_node.type == 'function_item':
                            method_info = self._extract_rust_function(item_node, content, code_bytes, type_name)
                            if method_info:
                                methods.append(method_info)

            # Extract docstring (Rust doc comment before type)
            docstring = self._extract_rust_docstring(node, content, code_bytes)

            return ClassInfo(
                name=type_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
                methods=methods,
                decorators=[]
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract Rust type: {e}")
            return None

    def _extract_rust_function(self, node, content: str, code_bytes: bytes, class_name: Optional[str] = None) -> Optional[FunctionInfo]:
        """Extract function information from Rust AST node"""
        try:
            # Get function name
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                    break

            if not func_name:
                return None

            # Extract parameters
            parameters = []
            return_type = None
            for child in node.children:
                if child.type == 'parameters':
                    for param_node in child.children:
                        if param_node.type == 'parameter':
                            param_text = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf-8')
                            parameters.append(param_text)
                elif child.type in ('primitive_type', 'type_identifier'):
                    # This might be the return type
                    return_type = code_bytes[child.start_byte:child.end_byte].decode('utf-8')

            # Extract docstring
            docstring = self._extract_rust_docstring(node, content, code_bytes)

            # Check if async
            is_async = any(child.type == 'async' for child in node.children)

            return FunctionInfo(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                is_async=is_async,
                is_method=class_name is not None,
                class_name=class_name,
                decorators=[],
                complexity=1
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract Rust function: {e}")
            return None

    def _extract_rust_docstring(self, node, content: str, code_bytes: bytes) -> Optional[str]:
        """Extract Rust doc comment before a node"""
        parent = node.parent
        if parent:
            prev_sibling = None
            for i, child in enumerate(parent.children):
                if child == node and i > 0:
                    prev_sibling = parent.children[i - 1]
                    break

            if prev_sibling and prev_sibling.type in ('line_comment', 'block_comment'):
                comment_text = code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf-8')
                # Clean up Rust doc comment markers
                if comment_text.startswith('///'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('//!'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
                elif comment_text.startswith('//'):
                    return comment_text[2:].strip()

        return None

    def _parse_csharp(self, file_path: str, content: str) -> Optional[FileParseResult]:
        """Parse C# file using tree-sitter"""
        from .tree_sitter_parser import TSParser

        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)

        try:
            # Initialize parser for C#
            ts_parser = TSParser(language='csharp')
            tree = ts_parser.parse(content)
            if not tree:
                return self._parse_generic(file_path, content, 'csharp')

            root_node = tree.root_node

            classes = []
            functions = []
            imports = []
            module_docstring = None

            # Extract module-level documentation
            if self.extract_docstrings:
                module_docstring = self._extract_csharp_module_docstring(content, root_node)

            # Extract using statements
            if self.extract_imports:
                imports = self._extract_csharp_imports(root_node, content)

            # Extract classes, interfaces, and methods
            self._extract_csharp_items(
                root_node, content, classes, functions
            )

            # Count lines
            lines = content.split("\n")
            total_lines = len(lines)
            code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("//", "/*", "*")))
            comment_lines = sum(1 for line in lines if line.strip().startswith(("//", "/*", "*")))

            return FileParseResult(
                file_path=file_path,
                language='csharp',
                classes=classes,
                functions=functions,
                imports=imports,
                module_docstring=clean_docstring(module_docstring) if module_docstring else None,
                total_lines=total_lines,
                code_lines=code_lines,
                comment_lines=comment_lines,
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            return self._parse_generic(file_path, content, 'csharp')

    def _extract_csharp_module_docstring(self, content: str, root_node) -> Optional[str]:
        """Extract module-level documentation from C# file"""
        code_bytes = content.encode('utf-8')
        for child in root_node.children:
            if child.type == 'comment':
                comment_text = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                # Clean up comment markers
                if comment_text.startswith('///'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('//'):
                    return comment_text[2:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()
        return None

    def _extract_csharp_imports(self, root_node, content: str) -> List[ImportInfo]:
        """Extract 'using' statements from C# AST"""
        imports = []
        code_bytes = content.encode('utf-8')

        def visit_node(node):
            if node.type == 'using_directive':
                # Extract using namespace
                using_text = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                using_text = using_text.replace('using ', '').replace(';', '').strip()
                imports.append(ImportInfo(
                    module=using_text,
                    names=['*'],
                    is_from=False,
                    line=node.start_point[0] + 1,
                    level=0
                ))

            for child in node.children:
                visit_node(child)

        visit_node(root_node)
        return imports

    def _extract_csharp_items(self, root_node, content: str, classes: List, functions: List):
        """Extract classes, interfaces, and methods from C# AST"""
        code_bytes = content.encode('utf-8')

        def visit_node(node, current_class=None):
            # C# classes, interfaces, structs
            if node.type in ('class_declaration', 'interface_declaration', 'struct_declaration'):
                class_info = self._extract_csharp_class(node, content, code_bytes)
                if class_info:
                    classes.append(class_info)
            # C# methods
            elif node.type == 'method_declaration':
                func_info = self._extract_csharp_method(node, content, code_bytes, current_class)
                if func_info:
                    functions.append(func_info)
            else:
                # Recursively visit children
                for child in node.children:
                    if node.type in ('class_declaration', 'interface_declaration', 'struct_declaration'):
                        visit_node(child, node)
                    else:
                        visit_node(child, current_class)

        visit_node(root_node)

    def _extract_csharp_class(self, node, content: str, code_bytes: bytes) -> Optional[ClassInfo]:
        """Extract class/interface/struct information from C# AST node"""
        try:
            # Get class name
            name_node = None
            for child in node.children:
                if child.type == 'identifier':
                    name_node = child
                    break

            if not name_node:
                return None

            class_name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            # Extract base classes/interfaces
            bases = []
            for child in node.children:
                if child.type == 'base_list':
                    for subchild in child.children:
                        if subchild.type in ('identifier', 'qualified_name'):
                            bases.append(code_bytes[subchild.start_byte:subchild.end_byte].decode('utf-8'))

            # Extract methods
            methods = []
            for child in node.children:
                if child.type == 'declaration_list':
                    for member_node in child.children:
                        if member_node.type == 'method_declaration':
                            method_info = self._extract_csharp_method(member_node, content, code_bytes, class_name)
                            if method_info:
                                methods.append(method_info)

            # Extract docstring (XML doc comment before class)
            docstring = self._extract_csharp_docstring(node, content, code_bytes)

            # Extract attributes (C# decorators)
            decorators = []
            for child in node.children:
                if child.type == 'attribute_list':
                    for attr_node in child.children:
                        if attr_node.type == 'attribute':
                            attr_text = code_bytes[attr_node.start_byte:attr_node.end_byte].decode('utf-8')
                            decorators.append(attr_text)

            return ClassInfo(
                name=class_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
                methods=methods,
                decorators=decorators
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract C# class: {e}")
            return None

    def _extract_csharp_method(self, node, content: str, code_bytes: bytes, class_name: Optional[str] = None) -> Optional[FunctionInfo]:
        """Extract method information from C# AST node"""
        try:
            # Get method name
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code_bytes[child.start_byte:child.end_byte].decode('utf-8')
                    break

            if not func_name:
                return None

            # Extract parameters
            parameters = []
            return_type = None
            for child in node.children:
                if child.type == 'parameter_list':
                    for param_node in child.children:
                        if param_node.type == 'parameter':
                            param_text = code_bytes[param_node.start_byte:param_node.end_byte].decode('utf-8')
                            parameters.append(param_text)
                elif child.type in ('predefined_type', 'identifier', 'qualified_name'):
                    # This might be the return type
                    if not return_type:  # Take the first type identifier as return type
                        return_type = code_bytes[child.start_byte:child.end_byte].decode('utf-8')

            # Extract docstring
            docstring = self._extract_csharp_docstring(node, content, code_bytes)

            # Check if async
            is_async = any(child.type == 'async' or code_bytes[child.start_byte:child.end_byte] == b'async' for child in node.children)

            # Extract attributes (C# decorators)
            decorators = []
            for child in node.children:
                if child.type == 'attribute_list':
                    for attr_node in child.children:
                        if attr_node.type == 'attribute':
                            attr_text = code_bytes[attr_node.start_byte:attr_node.end_byte].decode('utf-8')
                            decorators.append(attr_text)

            return FunctionInfo(
                name=func_name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                is_async=is_async,
                is_method=class_name is not None,
                class_name=class_name,
                decorators=decorators,
                complexity=1
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract C# method: {e}")
            return None

    def _extract_csharp_docstring(self, node, content: str, code_bytes: bytes) -> Optional[str]:
        """Extract XML doc comment before a C# node"""
        parent = node.parent
        if parent:
            prev_sibling = None
            for i, child in enumerate(parent.children):
                if child == node and i > 0:
                    prev_sibling = parent.children[i - 1]
                    break

            if prev_sibling and prev_sibling.type == 'comment':
                comment_text = code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf-8')
                # Clean up XML doc comment markers
                if comment_text.startswith('///'):
                    return comment_text[3:].strip()
                elif comment_text.startswith('//'):
                    return comment_text[2:].strip()
                elif comment_text.startswith('/*') and comment_text.endswith('*/'):
                    return comment_text[2:-2].strip()

        return None


    def _parse_generic(self, file_path: str, content: str, language: str) -> FileParseResult:
        """Generic parsing for unsupported languages"""
        # Strip markdown code fences if present
        content = self._strip_markdown_code_fences(content)
        
        lines = content.split("\n")
        total_lines = len(lines)
        
        # Simple heuristics for code vs comment lines
        code_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("//", "#", "/*", "*", "*/")):
                comment_lines += 1
            else:
                code_lines += 1
        
        return FileParseResult(
            file_path=file_path,
            language=language,
            classes=[],
            functions=[],
            imports=[],
            module_docstring=None,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
        )

