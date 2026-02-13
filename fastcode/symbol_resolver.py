"""
Symbol Resolver - Resolve symbol names to their definition IDs

This module implements Task 3.2: 符号解析器 (Symbol Resolver)
Given a code symbol name, find its definition ID using local and imported resolution strategies.
"""

import logging
from typing import Optional, Dict, List, Any
from .global_index_builder import GlobalIndexBuilder
from .module_resolver import ModuleResolver


class SymbolResolver:
    """
    Resolve symbol names to their definition IDs

    Implements two resolution strategies:
    1. Local: Check if current file defines the symbol
    2. Imported: Check imports and resolve to source file definitions
    """

    def __init__(self, global_index: GlobalIndexBuilder, module_resolver: ModuleResolver):
        """
        Initialize SymbolResolver with dependency injection

        Args:
            global_index: GlobalIndexBuilder instance for lookup tables
            module_resolver: ModuleResolver instance for cross-file resolution
        """
        self.global_index = global_index
        self.module_resolver = module_resolver
        self.logger = logging.getLogger(__name__)

    def resolve_symbol(self, symbol_name: str, current_file_id: str, imports: List[Dict[str, Any]]) -> Optional[str]:
        """
        Resolve a symbol name to its definition ID

        Resolution strategy:
        1. Local: Check current file's exports first
        2. Imported: Check imports and resolve to source definitions

        Args:
            symbol_name: Name of the symbol to resolve (e.g., "helper", "MyClass")
            current_file_id: File ID of the current file
            imports: List of import dictionaries from Task 1.3 extraction

        Returns:
            Definition ID if found, None otherwise
        """
        if not symbol_name or not current_file_id:
            return None

        # Strategy 1: Local resolution - check current file first
        local_result = self._resolve_local(symbol_name, current_file_id)
        if local_result:
            self.logger.debug(f"Resolved '{symbol_name}' locally to {local_result}")
            return local_result

        # Strategy 2: Imported resolution - check imports
        imported_result = self._resolve_imported(symbol_name, imports, current_file_id)
        if imported_result:
            self.logger.debug(f"Resolved '{symbol_name}' via import to {imported_result}")
            return imported_result

        self.logger.debug(f"Could not resolve '{symbol_name}'")
        return None

    def _resolve_local(self, symbol_name: str, current_file_id: str) -> Optional[str]:
        """
        Resolve symbol locally in current file

        Args:
            symbol_name: Symbol name to resolve
            current_file_id: Current file ID

        Returns:
            Definition ID if found locally, None otherwise
        """
        # Get current file's module path
        current_module_path = self._get_module_path_by_file_id(current_file_id)
        if not current_module_path:
            return None

        # Check if symbol is exported by current module
        return self.global_index.get_exported_symbol_id(current_module_path, symbol_name)

    def _resolve_imported(self, symbol_name: str, imports: List[Dict[str, Any]],
                        current_file_id: Optional[str] = None) -> Optional[str]:
        """
        Resolve symbol through imports

        Args:
            symbol_name: Symbol name to resolve
            imports: List of import dictionaries with structure:
                    [{'module': 'utils', 'names': ['helper'], 'alias': None, 'level': 0}, ...]
            current_file_id: Current file ID for getting module path

        Returns:
            Definition ID if found through imports, None otherwise
        """
        for import_info in imports:
            # Handle different import patterns
            if self._matches_import(symbol_name, import_info):
                # Resolve the module to get target file ID
                current_module_path = self._get_current_module_path_for_imports(current_file_id)
                target_file_id = self.module_resolver.resolve_import(
                    current_module_path=current_module_path or "",
                    import_name=import_info.get('module', ''),
                    level=import_info.get('level', 0)
                )

                if target_file_id:
                    target_module_path = self._get_module_path_by_file_id(target_file_id)
                    if target_module_path:
                        # For "from X import Y" pattern
                        if import_info.get('names'):
                            imported_names = import_info['names']
                            
                            # 1. Exact match (e.g. import func)
                            if symbol_name in imported_names:
                                return self.global_index.get_exported_symbol_id(target_module_path, symbol_name)

                            # 2. Alias match
                            alias = import_info.get('alias')
                            if alias and symbol_name == alias:
                                original_name = imported_names[0] if imported_names else symbol_name
                                return self.global_index.get_exported_symbol_id(target_module_path, original_name)
                            
                            # --- [CRITICAL FIX] Member match (Class.Method) ---
                            # Check if we are looking for "RepositoryLoader.load_from_url" 
                            # and we imported "RepositoryLoader"
                            for name in imported_names:
                                if symbol_name.startswith(name + '.'):
                                    # Look up the full symbol "RepositoryLoader.load_from_url" 
                                    # in the target module (fastcode.loader)
                                    return self.global_index.get_exported_symbol_id(target_module_path, symbol_name)
                            # --------------------------------------------------

                        # For "import X" pattern (remains same)
                        elif import_info.get('module') and symbol_name.startswith(import_info['module'] + '.'):
                            actual_symbol = symbol_name[len(import_info['module']) + 1:]
                            return self.global_index.get_exported_symbol_id(target_module_path, actual_symbol)

        return None

    def _matches_import(self, symbol_name: str, import_info: Dict[str, Any]) -> bool:
        """
        Check if symbol name matches this import statement
        """
        imported_names = import_info.get('names', [])
        alias = import_info.get('alias')
        module_name = import_info.get('module', '')

        # Direct name match: "from utils import helper" -> use "helper"
        if symbol_name in imported_names:
            return True

        # Alias match: "from utils import helper as h" -> use "h"
        if alias and symbol_name == alias:
            return True

        # Module prefix match: "import utils" -> use "utils.helper"
        if module_name and symbol_name.startswith(module_name + '.'):
            return True
            
        # --- [CRITICAL FIX] Check if symbol is a method of an imported class ---
        imported_names = import_info.get('names', [])
        for name in imported_names:
            if symbol_name.startswith(name + '.'):
                return True
        # -----------------------------------------------------------------------

        return False

    def _get_module_path_by_file_id(self, file_id: str) -> Optional[str]:
        """
        Get module path from file_id using reverse lookup

        Args:
            file_id: File ID to look up

        Returns:
            Module path if found, None otherwise
        """
        # Search module_map for this file_id
        for module_path, mapped_file_id in self.global_index.module_map.items():
            if mapped_file_id == file_id:
                return module_path
        return None

    def _get_current_module_path_for_imports(self, current_file_id: Optional[str] = None) -> str:
        """
        Get current module path for import resolution

        Args:
            current_file_id: Current file ID to look up module path

        Returns:
            Current module path string (empty string as fallback)
        """
        if current_file_id:
            return self._get_module_path_by_file_id(current_file_id) or ""
        return ""

    def get_resolution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about resolution performance

        Returns:
            Dictionary with resolution statistics
        """
        return {
            "modules_available": len(self.global_index.module_map),
            "exports_available": len(self.global_index.export_map),
            "files_mapped": len(self.global_index.file_map),
        }