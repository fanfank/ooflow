#!/usr/bin/env python3
"""
OoFlow setup script
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read version from ooflow/__init__.py
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "ooflow", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="ooflow",
    version=read_version(),
    author="fanfank",
    author_email="fanfank@example.com",
    description="A lightweight Python framework for building asynchronous data processing pipelines with stateful nodes",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fanfank/ooflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No external dependencies - uses only Python standard library
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "coverage>=5.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "coverage>=5.0",
        ],
    },
    keywords="async, ai, streaming, stateful, asyncio, graph, nodes, pipeline, workflow, data-processing",
    project_urls={
        "Bug Reports": "https://github.com/fanfank/ooflow/issues",
        "Source": "https://github.com/fanfank/ooflow",
        "Documentation": "https://github.com/fanfank/ooflow#readme",
    },
)
