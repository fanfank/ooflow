"""
OoFlow - A lightweight AI-Ready Python framework for building asynchronous data processing pipelines with stateful nodes.
"""

__version__ = "0.1.1"
__author__ = "fanfank"

from .ooflow import Node, Context, OoFlow, Edge, create, setup_logger, logger

__all__ = ["Node", "Context", "OoFlow", "Edge", "create", "setup_logger", "logger"]