"""
Import Extractor using Tree-sitter Query (S-expressions)
Extracts import information from Python code efficiently using pattern matching.
"""

from typing import List, Dict, Any, Optional
import logging
from tree_sitter import Node, Query, QueryCursor
from .tree_sitter_parser import TSParser


class ImportExtractor:
    """
    Extracts import information using Tree-sitter Queries.
    Depends on TSParser for underlying parsing logic.
    """

    IMPORT_QUERY_SCM = """
    (import_statement
        name: (_) @import.item)

    (import_from_statement
        name: (_) @from.item)

    (import_from_statement
        (wildcard_import) @from.item)
    """

    def __init__(self, parser: Optional[TSParser] = None):
        self.logger = logging.getLogger(__name__)
        self.ts_parser = parser or TSParser()

        if not self.ts_parser.is_healthy():
            raise RuntimeError("TSParser could not be initialized.")

        self.query = Query(self.ts_parser.language, self.IMPORT_QUERY_SCM)

    def extract_imports(self, code: str) -> List[Dict[str, Any]]:
        tree = self.ts_parser.parse(code)
        if not tree:
            return []

        root_node = tree.root_node
        imports = []

        from_stmt_cache = {}

        cursor = QueryCursor(self.query)

        captures = cursor.captures(root_node)

        for capture_name, nodes in captures.items():
            for node in nodes:
                # --- Case 1: Import (import os) ---
                if capture_name == 'import.item':
                    name, alias = self._parse_aliased_import(node, code)
                    imports.append({
                        'module': name,
                        'names': [name],
                        'alias': alias,
                        'level': 0
                    })

                # --- Case 2: From Import (from x import y) ---
                elif capture_name == 'from.item':
                    # Search upward for parent node import_from_statement
                    parent = node.parent
                    while parent and parent.type != 'import_from_statement':
                        parent = parent.parent

                    if not parent:
                        continue

                    # Parse the context of parent statement (module, level)
                    if parent.id not in from_stmt_cache:
                        from_stmt_cache[parent.id] = self._parse_from_context(parent, code)

                    module, level = from_stmt_cache[parent.id]

                    # Parse current import item
                    if node.type == 'wildcard_import':
                        name = '*'
                        alias = None
                    else:
                        name, alias = self._parse_aliased_import(node, code)

                    imports.append({
                        'module': module,
                        'names': [name],
                        'alias': alias,
                        'level': level
                    })

        return imports

    def _parse_aliased_import(self, node: Node, code: str) -> tuple[str, Optional[str]]:
        """Parse name or aliased_import"""
        def get_text(n):
            return code[n.start_byte:n.end_byte]

        if node.type == 'aliased_import':
            name_node = node.child_by_field_name('name')
            alias_node = node.child_by_field_name('alias')
            name = get_text(name_node) if name_node else ""
            alias = get_text(alias_node) if alias_node else None
            return name, alias

        # dotted_name or other identifiers
        return get_text(node), None

    def _parse_from_context(self, stmt_node: Node, code: str) -> tuple[str, int]:
        """
        Parse module and level of import_from_statement.
        Supports 'from . import x', 'from .utils import x', 'from utils import x'
        """
        level = 0
        module_name = ""

        # Directly traverse child nodes of statement to find relative_import or dotted_name
        for child in stmt_node.children:

            if child.type == 'relative_import':
                # relative_import node may contain dots and name, e.g., "..utils"
                # or only dots, e.g., ".."
                text = code[child.start_byte:child.end_byte]

                # Count number of dots
                current_dots = 0
                for char in text:
                    if char == '.':
                        current_dots += 1
                    else:
                        break
                level = current_dots

                # The remaining part is the module name
                module_name = text[current_dots:]

            elif child.type == 'dotted_name':
                # Absolute import, only process when relative_import is not encountered
                # (Prevent misidentifying dotted_name after import as module)
                if level == 0 and not module_name:
                    # We need to ensure this dotted_name is before the "import" keyword
                    # Simple check method: see if its next sibling node is the 'import' keyword?
                    # Or simpler: the structure of import_from_statement is fixed as "from" module "import" ...
                    # As long as it's before the import keyword.
                    # Simplified handling here: usually the module's dotted_name appears first
                    module_name = code[child.start_byte:child.end_byte]

            elif child.type == 'import':
                # Encountered import keyword, indicating module part parsing is complete
                break

        return module_name, level