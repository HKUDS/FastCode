"""
Call Extractor - Extract function calls using Tree-sitter with scope tracking
Provides precise function call extraction for call graph construction
"""

import logging
from typing import Dict, List, Any, Optional, Set
import tree_sitter
from tree_sitter import Query, QueryCursor

from .tree_sitter_parser import TSParser


class CallExtractor:
    """
    Extract function calls from Python code using Tree-sitter with scope tracking.

    This class provides precise function call extraction capabilities while maintaining
    context about which function/class contains each call - essential for building
    accurate call graphs.
    """

    def __init__(self, parser: Optional[TSParser] = None):
        """
        Initialize CallExtractor with optional TSParser dependency injection.

        Args:
            parser: Optional TSParser instance. If None, creates own instance.
        """
        self.logger = logging.getLogger(__name__)

        # Support dependency injection pattern (consistent with other extractors)
        if parser is not None:
            self.parser = parser
        else:
            self.parser = TSParser()

        # Built-in function names to filter out
        self._builtin_functions = self._get_builtin_functions()

        # Initialize queries (will be compiled when needed)
        self._call_query = None
        self._scope_query = None
        self._init_type_query = None
        self._init_queries()

    def _get_builtin_functions(self) -> Set[str]:
        """
        Get set of Python built-in function names to filter out.

        Returns:
            Set of built-in function names
        """
        # Python built-in functions that should be filtered out
        builtins = {
            # Standard built-ins
            'abs', 'all', 'any', 'bin', 'bool', 'breakpoint', 'bytearray',
            'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
            'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
            'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
            'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
            'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
            'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
            'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
            'super', 'tuple', 'type', 'vars', 'zip',

            # Common built-in types (when used as constructors)
            'type', 'isinstance', 'issubclass'
        }

        return builtins

    def _init_queries(self):
        """Initialize Tree-sitter queries for function call detection."""
        if not self.parser.is_healthy():
            self.logger.error("TSParser not healthy, cannot initialize queries")
            return

        language = self.parser.get_language()
        if language is None:
            self.logger.error("Language not available, cannot initialize queries")
            return

        try:
            # Use standard tree_sitter.Query constructor (not language.query)
            self._call_query = Query(language, """
                (call
                    function: (_) @function_name
                ) @call
            """)

            self._scope_query = Query(language, """
                (function_definition
                    name: (identifier) @func_name
                ) @func_def

                (class_definition
                    name: (identifier) @class_name
                ) @class_def
            """)

            # New query for instance variable type inference
            self._init_type_query = Query(language, """
                ; 1. Constructor assignment: self.variable = ClassName(...)
                (assignment
                    left: (attribute
                        object: (identifier) @self_obj
                        attribute: (identifier) @attr_name
                    )
                    right: (call
                        function: (identifier) @class_name
                    )
                ) @constructor_assign
                
                ; --- FIX START: Add support for local variables ---
                ; 4. Local variable assignment: variable = ClassName(...)
                (assignment
                    left: (identifier) @var_name
                    right: (call
                        function: (identifier) @class_name
                    )
                ) @local_constructor_assign
                ; --- FIX END ---

                ; 2. Type hint annotation: self.variable: ClassName
                ; Matches any node with 'left' and 'type' fields (handles version differences)
                (_
                    left: (attribute
                        object: (identifier) @self_obj_hint
                        attribute: (identifier) @attr_name_hint
                    )
                    type: (_) @type_annotation
                ) @type_hint

                ; 3. Type hint with assignment: self.variable: ClassName = ...
                (_
                    left: (attribute
                        object: (identifier) @self_obj_assign
                        attribute: (identifier) @attr_name_assign
                    )
                    type: (_) @type_annotation_assign
                    right: (_) @assign_value
                ) @type_hint_assign
            """)

            self.logger.debug("Tree-sitter queries initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize queries: {e}")
            self._call_query = None
            self._scope_query = None
            self._init_type_query = None

    def extract_calls(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract function calls from Python code with scope tracking.

        Args:
            code: Python source code string
            file_path: Path to the source file (for context)

        Returns:
            List of dictionaries with call information including scope context
        """
        if not self.parser.is_healthy():
            self.logger.error("TSParser not healthy, cannot extract calls")
            return []

        if self._call_query is None or self._scope_query is None:
            self.logger.error("Queries not initialized, cannot extract calls")
            return []

        # Parse the code
        tree = self.parser.parse(code)
        if tree is None:
            self.logger.error(f"Failed to parse code for {file_path}")
            return []

        # First pass: find all scopes (functions and classes)
        scopes = self._extract_scopes(tree)

        # Second pass: find all function calls and assign to scopes
        calls = self._extract_calls_with_scopes(tree, scopes, file_path)

        self.logger.debug(f"Extracted {len(calls)} calls from {file_path}")
        return calls

    def _execute_query(self, query: Query, node: tree_sitter.Node) -> List[Any]:
        """
        Execute Tree-sitter query and return captures in standardized format.

        Args:
            query: Tree-sitter Query object
            node: Root node for query execution

        Returns:
            List of (node, capture_name) tuples
        """
        try:
            cursor = QueryCursor(query)
            captures = cursor.captures(node)

            # Handle different tree-sitter versions
            # Older versions return dict, newer versions return list
            if isinstance(captures, dict):
                # Convert dict format to list of (node, capture_name) tuples
                result = []
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        result.append((node, capture_name))
                return result
            else:
                # Newer versions already return list format
                return captures

        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            return []

    def _extract_scopes(self, tree: tree_sitter.Tree) -> List[Dict[str, Any]]:
        """
        Extract function and class definitions for scope tracking.

        Args:
            tree: Tree-sitter syntax tree

        Returns:
            List of scope information with ranges
        """
        scopes = []

        # Execute scope query
        captures = self._execute_query(self._scope_query, tree.root_node)
        if not captures:
            return scopes

        # Process captures to build scope list
        for node, tag in captures:
            if tag == 'func_def':
                # Get function name from the name field
                func_name_node = node.child_by_field_name('name')
                if func_name_node:
                    scopes.append({
                        'type': 'function',
                        'name': func_name_node.text.decode('utf-8'),
                        'range': {
                            'start_byte': node.start_byte,
                            'end_byte': node.end_byte,
                            'start_point': node.start_point,
                            'end_point': node.end_point
                        }
                    })

            elif tag == 'class_def':
                # Get class name from the name field
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    scopes.append({
                        'type': 'class',
                        'name': class_name_node.text.decode('utf-8'),
                        'range': {
                            'start_byte': node.start_byte,
                            'end_byte': node.end_byte,
                            'start_point': node.start_point,
                            'end_point': node.end_point
                        }
                    })

        # Sort scopes by start position for proper nesting detection
        scopes.sort(key=lambda s: s['range']['start_byte'])

        return scopes

    def _extract_calls_with_scopes(self, tree: tree_sitter.Tree, scopes: List[Dict[str, Any]], file_path: str) -> List[Dict[str, Any]]:
        """
        Extract function calls and assign them to appropriate scopes.

        Args:
            tree: Tree-sitter syntax tree
            scopes: List of scope definitions from _extract_scopes
            file_path: Path to the source file

        Returns:
            List of call information with scope context
        """
        calls = []

        # Execute call query
        captures = self._execute_query(self._call_query, tree.root_node)
        if not captures:
            return calls

        # Process each call capture
        for node, tag in captures:
            if tag == 'call':
                call_info = self._process_call_node(node, scopes, file_path)
                if call_info and not self._should_filter_call(call_info):
                    calls.append(call_info)

        return calls

    def _process_call_node(self, call_node: tree_sitter.Node, scopes: List[Dict[str, Any]], file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single call node and extract call information.

        Args:
            call_node: Tree-sitter call node
            scopes: List of scope definitions
            file_path: Path to the source file

        Returns:
            Call information dictionary or None if extraction fails
        """
        # Get the function being called
        function_node = call_node.child_by_field_name('function')
        if function_node is None:
            return None

        # Determine the call type and extract names
        call_info = self._extract_call_details(function_node)
        if not call_info:
            return None

        # Find the scope this call belongs to
        scope_id = self._find_scope_for_call(call_node, scopes)

        # Build complete call information
        result = {
            'call_name': call_info['call_name'],
            'base_object': call_info.get('base_object'),
            'call_type': call_info['call_type'],  # 'simple', 'attribute', 'method'
            'scope_id': scope_id,
            'scope_type': self._get_scope_type_from_id(scope_id),
            'file_path': file_path,
            'range': {
                'start_byte': call_node.start_byte,
                'end_byte': call_node.end_byte,
                'start_point': call_node.start_point,
                'end_point': call_node.end_point
            },
            'node_text': call_node.text.decode('utf-8')
        }

        return result

    def _extract_call_details(self, function_node: tree_sitter.Node) -> Optional[Dict[str, Any]]:
        """
        Extract details about the function being called.

        Args:
            function_node: Tree-sitter node representing the function part of a call

        Returns:
            Dictionary with call details or None if extraction fails
        """
        if function_node.type == 'identifier':
            # Simple function call: func()
            call_name = function_node.text.decode('utf-8')
            return {
                'call_name': call_name,
                'call_type': 'simple'
            }

        elif function_node.type == 'attribute':
            # Attribute access: obj.method() or module.function()
            # Extract object and attribute names
            object_node = function_node.child_by_field_name('object')
            attribute_node = function_node.child_by_field_name('attribute')

            if object_node and attribute_node:
                base_object = object_node.text.decode('utf-8')
                call_name = attribute_node.text.decode('utf-8')

                return {
                    'call_name': call_name,
                    'base_object': base_object,
                    'call_type': 'attribute'
                }

        elif function_node.type == 'subscript':
            # Subscript access: obj[index]() - less common but possible
            # For now, skip this complexity
            return None

        # For other node types, skip
        return None

    def _find_scope_for_call(self, call_node: tree_sitter.Node, scopes: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find which scope (function/class) contains this call.

        Args:
            call_node: Tree-sitter node for the call
            scopes: List of scope definitions sorted by position

        Returns:
            Scope identifier in format "type::name" or None if no scope found
        """
        call_position = call_node.start_byte

        # Iterate scopes in reverse order (to find the innermost scope first)
        for scope in reversed(scopes):
            scope_range = scope['range']
            if scope_range['start_byte'] <= call_position < scope_range['end_byte']:
                # Found containing scope
                return f"{scope['type']}::{scope['name']}"

        # No containing scope found (module-level call)
        return None

    def _get_scope_type_from_id(self, scope_id: Optional[str]) -> Optional[str]:
        """
        Get scope type from scope ID.

        Args:
            scope_id: Scope identifier from _find_scope_for_call

        Returns:
            Scope type ('function', 'class') or None
        """
        if scope_id is None:
            return None

        # Extract type from scope_id (format "type::name")
        parts = scope_id.split('::', 1)
        if len(parts) == 2:
            return parts[0]

        return None

    def _should_filter_call(self, call_info: Dict[str, Any]) -> bool:
        """
        Determine if a call should be filtered out (e.g., built-in functions).

        Args:
            call_info: Call information dictionary

        Returns:
            True if the call should be filtered out, False otherwise
        """
        call_name = call_info['call_name']

        # Filter out built-in functions
        if call_name in self._builtin_functions:
            return True

        # Filter out calls to self/cls for method calls
        # REMOVED: Allow self/cls calls to be processed for graph construction
        # if call_info.get('base_object') in ['self', 'cls']:
        #     return True

        # Could add more filtering logic here
        return False

    def get_extraction_stats(self, total_calls: int, filtered_calls: int) -> Dict[str, Any]:
        """
        Get statistics about the extraction process.

        Args:
            total_calls: Total number of calls found
            filtered_calls: Number of calls filtered out

        Returns:
            Statistics dictionary
        """
        return {
            'builtin_functions_count': len(self._builtin_functions),
            'total_calls_found': total_calls,
            'calls_filtered': filtered_calls,
            'calls_kept': total_calls - filtered_calls,
            'filter_rate': filtered_calls / total_calls if total_calls > 0 else 0
        }

    def extract_instance_types(self, code: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract potential class types for instance variables from Python code.

        This method identifies patterns like:
        - self.variable = ClassName(...)  (constructor assignment)
        - self.variable: ClassName       (type hint)
        - self.variable: ClassName = ... (type hint with assignment)

        Args:
            code: Python source code string

        Returns:
            Dictionary mapping variable names to lists of potential class names
            Example: {'loader': ['RepositoryLoader', 'FileSystemLoader'], 'db': ['Postgres']}
        """
        if not self.parser.is_healthy():
            self.logger.error("TSParser not healthy, cannot extract instance types")
            return {}

        if self._init_type_query is None:
            self.logger.error("Type inference query not initialized, cannot extract instance types")
            return {}

        # Parse the code
        tree = self.parser.parse(code)
        if tree is None:
            self.logger.error("Failed to parse code for instance type extraction")
            return {}

        # 1. Extract scopes first to know where we are
        scopes = self._extract_scopes(tree)
        
        # Initialize result with a 'global' scope
        # structure: { "function::name": { "var": ["Type"] } }
        scoped_types: Dict[str, Dict[str, List[str]]] = {}

        captures = self._execute_query(self._init_type_query, tree.root_node)
        
        for node, tag in captures:
            # Find which scope this assignment belongs to
            scope_id = self._find_scope_for_call(node, scopes) or "global"
            
            if scope_id not in scoped_types:
                scoped_types[scope_id] = {}
                
            # Pass the specific scope dict to the processors
            target_dict = scoped_types[scope_id]
            
            try:
                if tag == 'constructor_assign':
                    self._process_constructor_assignment(node, target_dict)
                elif tag == 'local_constructor_assign':
                    self._process_local_constructor_assignment(node, target_dict)
                elif tag == 'type_hint':
                    self._process_type_hint(node, target_dict)
                elif tag == 'type_hint_assign':
                    self._process_type_hint_assignment(node, target_dict)
            except Exception:
                continue

        return scoped_types

    def _process_constructor_assignment(self, node: tree_sitter.Node, instance_types: Dict[str, List[str]]):
        """
        Process constructor assignment pattern: self.variable = ClassName(...)

        Args:
            node: Tree-sitter node for the assignment
            instance_types: Dictionary to store extracted types
        """
        # Get the left side (attribute) to extract variable name (support both 'left' and 'target' field names)
        left_node = node.child_by_field_name('left')
        if not left_node:
            left_node = node.child_by_field_name('target')
        if not left_node or left_node.type != 'attribute':
            return

        # Extract variable name from attribute
        attr_node = left_node.child_by_field_name('attribute')
        if not attr_node:
            return

        # --- FIX: Capture full name (e.g., 'self.loader') ---
        var_name = attr_node.text.decode('utf-8')
        obj_node = left_node.child_by_field_name('object')
        if obj_node:
            obj_name = obj_node.text.decode('utf-8')
            var_name = f"{obj_name}.{var_name}"
        # ----------------------------------------------------

        # Get the right side (call) to extract class name
        right_node = node.child_by_field_name('right')
        if not right_node or right_node.type != 'call':
            return

        # Extract class name from the function being called
        func_node = right_node.child_by_field_name('function')
        if not func_node or func_node.type != 'identifier':
            return

        class_name = func_node.text.decode('utf-8')

        # Add to instance_types dictionary
        if var_name not in instance_types:
            instance_types[var_name] = []
        instance_types[var_name].append(class_name)

    def _process_type_hint(self, node: tree_sitter.Node, instance_types: Dict[str, List[str]]):
        """
        Process type hint pattern: self.variable: ClassName

        Args:
            node: Tree-sitter node for the annotated assignment
            instance_types: Dictionary to store extracted types
        """
        # Extract variable name from target (support both 'target' and 'left' field names)
        target_node = node.child_by_field_name('target')
        if not target_node:
            target_node = node.child_by_field_name('left')
        if not target_node or target_node.type != 'attribute':
            return

        attr_node = target_node.child_by_field_name('attribute')
        if not attr_node:
            return

        # --- FIX: Capture full name ---
        var_name = attr_node.text.decode('utf-8')
        obj_node = target_node.child_by_field_name('object')
        if obj_node:
            obj_name = obj_node.text.decode('utf-8')
            var_name = f"{obj_name}.{var_name}"
        # ------------------------------

        # Extract class name from type annotation (support both 'annotation' and 'type' field names)
        annotation_node = node.child_by_field_name('annotation')
        if not annotation_node:
            annotation_node = node.child_by_field_name('type')
        if not annotation_node:
            return

        class_name = self._extract_type_from_annotation(annotation_node)
        if class_name:
            if var_name not in instance_types:
                instance_types[var_name] = []
            instance_types[var_name].append(class_name)

    def _process_type_hint_assignment(self, node: tree_sitter.Node, instance_types: Dict[str, List[str]]):
        """
        Process type hint with assignment pattern: self.variable: ClassName = ...

        Args:
            node: Tree-sitter node for the annotated assignment
            instance_types: Dictionary to store extracted types
        """
        # Extract variable name from target (support both 'target' and 'left' field names)
        target_node = node.child_by_field_name('target')
        if not target_node:
            target_node = node.child_by_field_name('left')
        if not target_node or target_node.type != 'attribute':
            return

        attr_node = target_node.child_by_field_name('attribute')
        if not attr_node:
            return

        var_name = attr_node.text.decode('utf-8')

        # Extract class name from type annotation (support both 'annotation' and 'type' field names)
        annotation_node = node.child_by_field_name('annotation')
        if not annotation_node:
            annotation_node = node.child_by_field_name('type')
        if not annotation_node:
            return

        class_name = self._extract_type_from_annotation(annotation_node)
        if class_name:
            if var_name not in instance_types:
                instance_types[var_name] = []
            instance_types[var_name].append(class_name)

    def _extract_type_from_annotation(self, annotation_node: tree_sitter.Node) -> Optional[str]:
        """
        Extract class name from type annotation node.

        Args:
            annotation_node: Tree-sitter node representing the type annotation

        Returns:
            Class name if extractable, None otherwise
        """
        # Handle simple identifier: ClassName
        if annotation_node.type == 'identifier':
            return annotation_node.text.decode('utf-8')

        # Handle type wrapper node (common in newer tree-sitter-python versions)
        # This fixes the bug where 'type' nodes containing the identifier were being ignored
        elif annotation_node.type == 'type':
            # Iterate through children to find the meaningful type content
            for child in annotation_node.children:
                # Recurse to handle identifier, attribute, or other nested structures
                result = self._extract_type_from_annotation(child)
                if result:
                    return result

        # Handle attribute access: module.ClassName
        elif annotation_node.type == 'attribute':
            # Get the last attribute in the chain
            attr_node = annotation_node.child_by_field_name('attribute')
            if attr_node and attr_node.type == 'identifier':
                return attr_node.text.decode('utf-8')

        # Handle other complex types (for future enhancement)
        # Could handle generics like List[ClassName], Optional[ClassName], etc.

        return None
    
    def _process_local_constructor_assignment(self, node: tree_sitter.Node, instance_types: Dict[str, List[str]]):
        """
        Process local assignment pattern: variable = ClassName(...)
        """
        # Extract variable name
        left_node = node.child_by_field_name('left') or node.child_by_field_name('target')
        if not left_node or left_node.type != 'identifier':
            return
            
        var_name = left_node.text.decode('utf-8')
        
        # Extract class name from the function being called
        right_node = node.child_by_field_name('right')
        if not right_node or right_node.type != 'call':
            return

        func_node = right_node.child_by_field_name('function')
        if not func_node or func_node.type != 'identifier':
            return
            
        class_name = func_node.text.decode('utf-8')
        
        # Add to types dictionary
        if var_name not in instance_types:
            instance_types[var_name] = []
        instance_types[var_name].append(class_name)