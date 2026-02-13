"""
Global Index Builder - Build global lookup maps for code elements

This module implements Task 2.1: 构建 FileID Map 和 Module Map
Creates lookup tables to resolve "who在哪" (who is where) problems.
"""

import logging
from typing import Dict, List, Optional, Any
import os

from .indexer import CodeElement
from .path_utils import file_path_to_module_path, normalize_repo_root


class GlobalIndexBuilder:
    """
    Build global lookup maps for resolving code element locations

    Creates three main maps:
    - file_map: abs_path -> file_id
    - module_map: dotted.path -> file_id
    - export_map: module_dotted_path -> {symbol_name: node_id}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GlobalIndexBuilder

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Main lookup maps
        self.file_map: Dict[str, str] = {}      # abs_path -> file_id
        self.module_map: Dict[str, str] = {}    # dotted.module.path -> file_id
        self.export_map: Dict[str, Dict[str, str]] = {}  # module_path -> {symbol_name: node_id}

        # Statistics
        self.stats = {
            "files_processed": 0,
            "modules_created": 0,
            "symbols_exported": 0,
            "errors": 0
        }

    def build_maps(self, elements: List[CodeElement], repo_root: str) -> None:
        """
        Build file_map, module_map, and export_map from indexed elements

        Args:
            elements: List of CodeElement objects from indexer
            repo_root: Repository root directory for module path conversion
        """
        self.logger.info(f"Building global maps from {len(elements)} elements")
        self.logger.info(f"Repository root: {repo_root}")

        # Reset maps and stats
        self.file_map.clear()
        self.module_map.clear()
        self.export_map.clear()
        self.stats = {"files_processed": 0, "modules_created": 0, "symbols_exported": 0, "errors": 0}

        # Normalize repo root
        norm_repo_root = normalize_repo_root(repo_root)

        # Process only file-level elements
        file_elements = [elem for elem in elements if elem.type == "file"]

        for element in file_elements:
            try:
                self._process_file_element(element, norm_repo_root)
                self.stats["files_processed"] += 1
            except Exception as e:
                self.logger.error(f"Error processing file {element.file_path}: {e}")
                self.stats["errors"] += 1

        self.logger.info(
            f"Built maps: {len(self.file_map)} file paths, "
            f"{len(self.module_map)} module paths"
        )

        # Build export symbol map from class and function elements
        self._build_export_symbol_map(elements)

        self.logger.info(
            f"Built maps: {len(self.file_map)} file paths, "
            f"{len(self.module_map)} module paths, "
            f"{len(self.export_map)} modules with exported symbols"
        )

        if self.stats["errors"] > 0:
            self.logger.warning(f"Encountered {self.stats['errors']} errors during processing")

    def _process_file_element(self, element: CodeElement, repo_root: str) -> None:
        """
        Process a single file element and add to maps

        Args:
            element: CodeElement of type "file"
            repo_root: Normalized repository root path
        """
        # Add to file_map: abs_path -> file_id
        abs_path = os.path.abspath(element.file_path)
        self.file_map[abs_path] = element.id

        # Convert to module path and add to module_map
        module_path = file_path_to_module_path(element.file_path, repo_root)

        if module_path:
            self.module_map[module_path] = element.id
            self.stats["modules_created"] += 1
            self.logger.debug(f"Mapped module '{module_path}' -> {element.id}")
        else:
            self.logger.debug(f"Could not create module path for {element.file_path}")

    def get_file_id_by_path(self, abs_path: str) -> Optional[str]:
        """
        Get file_id from absolute file path

        Args:
            abs_path: Absolute file path

        Returns:
            File ID if found, None otherwise
        """
        # Normalize the path for consistent lookup
        norm_path = os.path.abspath(abs_path)
        return self.file_map.get(norm_path)

    def get_file_id_by_module(self, module_path: str) -> Optional[str]:
        """
        Get file_id from dotted module path

        Args:
            module_path: Dotted module path (e.g., "app.services.auth")

        Returns:
            File ID if found, None otherwise
        """
        return self.module_map.get(module_path)

    def get_all_file_ids(self) -> List[str]:
        """
        Get all file IDs

        Returns:
            List of all file IDs
        """
        return list(self.file_map.values())

    def get_all_modules(self) -> List[str]:
        """
        Get all module paths

        Returns:
            List of all module paths
        """
        return list(self.module_map.keys())

    def contains_file(self, abs_path: str) -> bool:
        """
        Check if a file path exists in the file_map

        Args:
            abs_path: Absolute file path

        Returns:
            True if file exists in map, False otherwise
        """
        norm_path = os.path.abspath(abs_path)
        return norm_path in self.file_map

    def contains_module(self, module_path: str) -> bool:
        """
        Check if a module path exists in the module_map

        Args:
            module_path: Dotted module path

        Returns:
            True if module exists in map, False otherwise
        """
        return module_path in self.module_map

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the built maps

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "file_map_size": len(self.file_map),
            "module_map_size": len(self.module_map),
            "module_coverage": self.stats["modules_created"] / max(1, self.stats["files_processed"])
        }

    def validate_maps(self) -> List[str]:
        """
        Validate the consistency of built maps

        Returns:
            List of validation errors (empty if maps are valid)
        """
        errors = []

        # Check if all file_map values are in module_map values (except those without modules)
        file_ids_from_file_map = set(self.file_map.values())
        file_ids_from_module_map = set(self.module_map.values())

        # Files in module_map should also be in file_map
        orphaned_modules = file_ids_from_module_map - file_ids_from_file_map
        if orphaned_modules:
            errors.append(f"Modules without files: {orphaned_modules}")

        # Check for duplicate module paths
        module_to_files = {}
        for module_path, file_id in self.module_map.items():
            if module_path in module_to_files:
                errors.append(f"Duplicate module '{module_path}' for files {module_to_files[module_path]} and {file_id}")
            else:
                module_to_files[module_path] = file_id

        return errors

    def _build_export_symbol_map(self, elements: List[CodeElement]) -> None:
        """
        Build export symbol map from class and function elements
        Maps: module_dotted_path -> {symbol_name: node_id}
        """
        self.logger.info("Building export symbol map from class and function elements")

        # Filter class and function elements
        symbol_elements = [elem for elem in elements if elem.type in ("class", "function")]

        for element in symbol_elements:
            try:
                # Get module path for this element's file
                module_path = self._get_module_path_for_element(element)

                if module_path:
                    # Initialize module entry if not exists
                    if module_path not in self.export_map:
                        self.export_map[module_path] = {}

                    # Add symbol to export map
                    self.export_map[module_path][element.name] = element.id
                    
                    # --- [CRITICAL FIX] Export Class.Method for methods ---
                    if element.type == 'function':
                        class_name = element.metadata.get('class_name')
                        if class_name:
                            full_name = f"{class_name}.{element.name}"
                            self.export_map[module_path][full_name] = element.id
                            self.logger.debug(f"Exported method: {full_name}")
                    # ------------------------------------------------------
                    
                    self.stats["symbols_exported"] += 1

            except Exception as e:
                self.logger.error(f"Error processing symbol {element.name}: {e}")
                self.stats["errors"] += 1

        self.logger.info(f"Built export map with {self.stats['symbols_exported']} symbols from {len(self.export_map)} modules")

    def _get_module_path_for_element(self, element: CodeElement) -> Optional[str]:
        """
        Get module path for a code element

        Args:
            element: CodeElement object

        Returns:
            Module path (dotted notation) or None if not found
        """
        # Find the file_id for this element's file
        abs_path = os.path.abspath(element.file_path)
        file_id = self.file_map.get(abs_path)

        if not file_id:
            self.logger.warning(f"No file_id found for {element.file_path}")
            return None

        # Find the module path for this file_id
        for module_path, module_file_id in self.module_map.items():
            if module_file_id == file_id:
                return module_path

        return None

    def get_exported_symbol_id(self, module_path: str, symbol_name: str) -> Optional[str]:
        """
        Get the node ID of an exported symbol

        Args:
            module_path: Dotted module path (e.g., "app.models")
            symbol_name: Symbol name (e.g., "User")

        Returns:
            Node ID if found, None otherwise
        """
        return self.export_map.get(module_path, {}).get(symbol_name)

    def get_module_exports(self, module_path: str) -> Dict[str, str]:
        """
        Get all exported symbols for a module

        Args:
            module_path: Dotted module path

        Returns:
            Dictionary mapping symbol names to node IDs
        """
        return self.export_map.get(module_path, {}).copy()

    def clear(self) -> None:
        """Clear all maps and reset stats"""
        self.file_map.clear()
        self.module_map.clear()
        self.export_map.clear()
        self.stats = {"files_processed": 0, "modules_created": 0, "symbols_exported": 0, "errors": 0}