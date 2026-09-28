"""
conftest.py
------------
Ensures tests can find data/ and models/ (referenced with relative
paths like "data/raw/students.csv") regardless of the directory
`pytest` was invoked from.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_repo_root():
    original_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(original_cwd)
