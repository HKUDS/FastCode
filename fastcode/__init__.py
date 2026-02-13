"""
FastCode 2.0 - Repository-Level Code Understanding System
With Multi-Repository Support
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

from .main import FastCode
from .loader import RepositoryLoader
from .parser import CodeParser
from .indexer import CodeIndexer
from .retriever import HybridRetriever
from .answer_generator import AnswerGenerator
from .repo_overview import RepositoryOverviewGenerator
from .repo_selector import RepositorySelector
from .iterative_agent import IterativeAgent
from .agent_tools import AgentTools

__version__ = "2.0.0"
FastCode = FastCode

__all__ = [
    "FastCode",
    "FastCode",
    "RepositoryLoader",
    "CodeParser",
    "CodeIndexer",
    "HybridRetriever",
    "AnswerGenerator",
    "RepositoryOverviewGenerator",
    "RepositorySelector",
    "IterativeAgent",
    "AgentTools",
]

