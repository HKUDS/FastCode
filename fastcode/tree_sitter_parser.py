"""
Tree-sitter Parser Wrapper
Provides a simple interface for parsing code with tree-sitter
"""

import tree_sitter
from tree_sitter import Language, Parser
from typing import Optional, Dict
import logging


class TSParser:
    """
    Tree-sitter parser wrapper for multiple languages

    Provides a simple interface to initialize and use tree-sitter
    for parsing code into syntax trees for various programming languages.
    """

    def __init__(self, language: str = 'python'):
        """
        Initialize the tree-sitter parser for a specific language

        Args:
            language: Programming language to parse ('python', 'javascript', 'typescript', etc.)
        """
        self.logger = logging.getLogger(__name__)
        self.current_language_name = language.lower()
        self.parser = None
        self.language = None
        self.languages_cache: Dict[str, Language] = {}  # Cache loaded languages
        self._initialize_parser()

    def _initialize_parser(self):
        """Initialize tree-sitter parser and language"""
        try:
            # Load the specified language
            self.language = self._load_language(self.current_language_name)

            # Initialize the parser with the language
            self.parser = Parser(self.language)

            self.logger.debug(f"Tree-sitter {self.current_language_name} parser initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize tree-sitter parser for {self.current_language_name}: {e}")
            raise

    def _load_language(self, language_name: str) -> Language:
        """
        Load a tree-sitter language

        Args:
            language_name: Name of the language to load

        Returns:
            Language object
        """
        # Check cache first
        if language_name in self.languages_cache:
            return self.languages_cache[language_name]

        # Import the appropriate tree-sitter language module
        if language_name == 'python':
            import tree_sitter_python
            lang = Language(tree_sitter_python.language())
        elif language_name == 'javascript':
            import tree_sitter_javascript
            lang = Language(tree_sitter_javascript.language())
        elif language_name == 'typescript':
            import tree_sitter_typescript
            # TypeScript has both typescript and tsx
            lang = Language(tree_sitter_typescript.language_typescript())
        elif language_name == 'tsx':
            import tree_sitter_typescript
            lang = Language(tree_sitter_typescript.language_tsx())
        elif language_name == 'c':
            import tree_sitter_c
            lang = Language(tree_sitter_c.language())
        elif language_name == 'cpp':
            import tree_sitter_cpp
            lang = Language(tree_sitter_cpp.language())
        elif language_name == 'rust':
            import tree_sitter_rust
            lang = Language(tree_sitter_rust.language())
        elif language_name == 'csharp':
            import tree_sitter_c_sharp
            lang = Language(tree_sitter_c_sharp.language())
        elif language_name == 'java':
            import tree_sitter_java
            lang = Language(tree_sitter_java.language())
        elif language_name == 'go':
            import tree_sitter_go
            lang = Language(tree_sitter_go.language())
        else:
            raise ValueError(f"Unsupported language: {language_name}")

        # Cache the language
        self.languages_cache[language_name] = lang
        return lang

    def set_language(self, language_name: str):
        """
        Switch the parser to a different language

        Args:
            language_name: Name of the language to switch to
        """
        try:
            self.current_language_name = language_name.lower()
            self.language = self._load_language(self.current_language_name)
            self.parser.set_language(self.language)
            self.logger.debug(f"Switched parser to {self.current_language_name}")
        except Exception as e:
            self.logger.error(f"Failed to switch language to {language_name}: {e}")
            raise

    def parse(self, code: str, language: Optional[str] = None) -> Optional[tree_sitter.Tree]:
        """
        Parse code string into a tree-sitter syntax tree

        Args:
            code: Source code string to parse
            language: Optional language override (will switch parser if different)

        Returns:
            Parsed syntax tree or None if parsing failed
        """
        # Switch language if requested
        if language and language.lower() != self.current_language_name:
            self.set_language(language)

        if not self.is_healthy():
            self.logger.error("Parser not properly initialized")
            return None

        if code is None or not isinstance(code, str):
            self.logger.warning("Invalid code input: code must be a string")
            return None

        try:
            # Convert code to bytes for tree-sitter
            code_bytes = code.encode('utf-8')

            # Parse the code
            tree = self.parser.parse(code_bytes)

            return tree

        except Exception as e:
            self.logger.error(f"Failed to parse code: {e}")
            return None

    def get_language(self) -> Optional[Language]:
        """Get the tree-sitter language object"""
        return self.language

    def is_healthy(self) -> bool:
        """Check if parser is properly initialized and ready"""
        return self.parser is not None and self.language is not None


