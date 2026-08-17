#!/usr/bin/env python3
"""Setup script for installing all llm-utils packages in editable mode."""

from setuptools import setup, find_namespace_packages
import os
from pathlib import Path

# Get all package directories
packages_dir = Path(__file__).parent / "packages"
package_dirs = [d for d in packages_dir.iterdir() if d.is_dir() and (d / "pyproject.toml").exists()]

# Collect all packages from all directories
all_packages = []
package_dir = {}

for pkg_dir in package_dirs:
    src_dir = pkg_dir / "src"
    if src_dir.exists():
        # Find packages in the src directory
        for root, dirs, files in os.walk(src_dir):
            if "__init__.py" in files:
                # Convert path to package name
                rel_path = os.path.relpath(root, src_dir)
                if rel_path == ".":
                    continue
                package_name = rel_path.replace(os.sep, ".")
                all_packages.append(package_name)
                
                # Build package_dir entry dynamically
                rel_root = os.path.relpath(root, Path(__file__).parent)
                package_dir[package_name] = rel_root

# Required dependencies
install_requires = [
    "fastapi>=0.116.1",
    "uvicorn>=0.35.0",
    "chardet>=5.2.0",
    "python-dotenv",
    "pydantic",
    "httpx",
    "langchain-core",
    "pyyaml",
]

setup(
    name="llm-utils-workspace",
    version="0.1.0",
    description="LLM utilities workspace with all packages",
    packages=all_packages,
    package_dir=package_dir,
    install_requires=install_requires,
    python_requires=">=3.9",
    zip_safe=False,
)