"""
Definition Extractor using Tree-sitter Query
Extracts class and function definitions with positions and parent relationships.
"""

from typing import List, Dict, Any, Optional
import logging
import os
import re
from tree_sitter import Node, Query, QueryCursor
from .tree_sitter_parser import TSParser


class DefinitionExtractor:
    """
    Extracts class and function definitions using Tree-sitter Queries.
    Supports both sync and async functions, with parent relationship tracking.
    """

    # Query for function and class definitions - only capture whole definition nodes
    DEFINITION_QUERY_SCM = """
    (function_definition) @function.def
    (class_definition) @class.def
    """

    def __init__(self, parser: Optional[TSParser] = None):
        self.logger = logging.getLogger(__name__)
        self.ts_parser = parser or TSParser()

        if not self.ts_parser.is_healthy():
            raise RuntimeError("TSParser could not be initialized.")

        self.query = Query(self.ts_parser.language, self.DEFINITION_QUERY_SCM)

    def extract_definitions(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract all class and function definitions from the given code.

        Args:
            code: Python source code string
            file_path: Absolute path to the source file

        Returns:
            List of dictionaries with definition information
        """
        tree = self.ts_parser.parse(code)
        if not tree:
            return []

        root_node = tree.root_node
        definitions = []

        cursor = QueryCursor(self.query)
        captures = cursor.captures(root_node)  # Returns Dict[str, List[Node]]

        # Process all captured nodes - handle both old and new API
        if isinstance(captures, dict):
            # Old API
            for capture_name, nodes in captures.items():
                for node in nodes:
                    try:
                        definition = self._process_definition_node(node, capture_name, code, file_path)
                        if definition:
                            definitions.append(definition)
                    except Exception as e:
                        self.logger.warning(f"Failed to process definition node: {e}")
                        continue
        else:
            # New API (tree-sitter 0.25+): List[Tuple[Node, str]]
            for node, capture_name in captures:
                try:
                    definition = self._process_definition_node(node, capture_name, code, file_path)
                    if definition:
                        definitions.append(definition)
                except Exception as e:
                    self.logger.warning(f"Failed to process definition node: {e}")
                    continue

        return definitions

    def _process_definition_node(self, node: Node, capture_name: str, code: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single definition node and extract information."""

        # Determine definition type and get name node
        if capture_name == 'function.def':
            # Use robust text-based async detection (expert-recommended approach)
            def_type = 'async_function' if self._is_async_function(node, code) else 'function'
            name_node = node.child_by_field_name('name')
        elif capture_name == 'class.def':
            def_type = 'class'
            name_node = node.child_by_field_name('name')
        else:
            return None

        if not name_node:
            return None

        # Extract basic information
        name = code[name_node.start_byte:name_node.end_byte]

        # Extract inheritance bases for classes
        bases = []
        if def_type == 'class':
            bases = self._extract_class_bases(node, code)

        # Find parent scope using backtracking (pass code for name extraction)
        parent = self._find_parent_scope(node, code)

        # Extract header using proper AST positioning (safe for type hints and strings)
        header = self._extract_definition_header_safe(node, code)

        # Generate unique ID
        relative_path = os.path.relpath(file_path)
        unique_id = f"{relative_path}::{parent}::{name}" if parent else f"{relative_path}::{name}"

        # Extract range information
        range_info = {
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_point": (node.start_point.row, node.start_point.column),
            "end_point": (node.end_point.row, node.end_point.column)
        }

        return {
            "id": unique_id,
            "name": name,
            "type": def_type,
            "range": range_info,
            "parent": parent,
            "header": header,
            "file_path": file_path,
            "bases": bases
        }

    def _is_async_function(self, node: Node, code: str) -> bool:
        """
        Check if a function_definition node represents an async function.
        Uses robust text-based checking that handles even weird formatting.
        """
        # Strategy 1: Check if the function node itself starts with async
        limit = min(node.end_byte, node.start_byte + 50)
        header_text = code[node.start_byte:limit]
        if bool(re.match(r'^async\b', header_text)):
            return True

        # Strategy 2: For weird formatting cases, check preceding context
        # Look at a reasonable range before the function for the async keyword
        search_start = max(0, node.start_byte - 100)  # Look back up to 100 chars
        preceding_context = code[search_start:node.start_byte]

        # Check if there's an 'async' keyword before our function definition
        # We use \basync\b to ensure it's a standalone async keyword
        return bool(re.search(r'\basync\b\s*$', preceding_context.strip()))

    def _find_parent_scope(self, node: Node, code: str) -> Optional[str]:
        """
        Find parent class scope using upward backtracking (O(Depth) complexity).

        Args:
            node: The definition node
            code: Original source code for text extraction

        Returns:
            Parent class name if this is a method, None otherwise
        """
        current = node.parent

        while current and current.type != 'module':
            if current.type == 'class_definition':
                # Found parent class, extract its name
                name_node = current.child_by_field_name('name')
                if name_node:
                    return code[name_node.start_byte:name_node.end_byte]
            current = current.parent

        return None

    def _extract_definition_header_safe(self, node: Node, code: str) -> str:
        """
        Extract definition header using AST-safe method.

        Instead of naively searching for ':', we find the 'body' node of the function/class
        and extract everything before it. This safely handles type hints and strings.
        """
        # Find the body node (if exists)
        body_node = node.child_by_field_name('body')

        if body_node:
            # Header is everything from node start to body start
            header_end = body_node.start_byte
        else:
            # Fallback: find the colon that marks end of header
            header_end = node.start_byte
            while (header_end < node.end_byte and
                   header_end < len(code) and
                   code[header_end] != ':'):
                header_end += 1

            if header_end < len(code) and code[header_end] == ':':
                header_end += 1  # Include the colon

        header = code[node.start_byte:header_end].strip()

        # Clean up extra whitespace and newlines
        lines = [line.strip() for line in header.split('\n') if line.strip()]

        # Return the most substantial line (usually the one with the actual definition)
        if len(lines) == 1:
            return lines[0]
        elif lines:
            # Find the line that contains the actual definition
            for line in lines:
                if any(keyword in line for keyword in ['def ', 'async def ', 'class ']):
                    return line
            return lines[0]  # Fallback

        return header

    def _extract_class_bases(self, class_node: Node, code: str) -> List[str]:
        """
        Extract base class names from a class definition node.

        Args:
            class_node: Tree-sitter class_definition node
            code: Source code string

        Returns:
            List of base class names
        """
        bases = []

        # Get the argument list (base classes) from the class definition
        # In tree-sitter, base classes are stored as child nodes
        for child in class_node.children:
            if child.type == 'argument_list':
                # Extract base class names from the argument list
                for base_child in child.children:
                    if base_child.type == 'identifier':
                        # Simple base class name like "BaseModel"
                        base_name = code[base_child.start_byte:base_child.end_byte]
                        bases.append(base_name)
                    elif base_child.type == 'attribute':
                        # Qualified base class name like "models.BaseModel"
                        # Look for identifier child within the attribute
                        for attr_child in base_child.children:
                            if attr_child.type == 'identifier':
                                attr_name = code[attr_child.start_byte:attr_child.end_byte]
                                # For simplicity, just take the last identifier (the class name)
                                # The full qualified name would require more complex handling
                                bases.append(attr_name)

        return bases