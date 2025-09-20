"""
OoFlow - A lightweight Python framework for building asynchronous data processing pipelines with stateful nodes.
"""

__version__ = "0.1.0"
__author__ = "fanfank"
__email__ = "fanfank@example.com"

from .ooflow import Node, Context, OoFlow, create, setup_logger

__all__ = ["Node", "Context", "OoFlow", "create", "setup_logger"]