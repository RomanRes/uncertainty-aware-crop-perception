# tests/conftest.py
"""
This file is used by pytest to configure the test environment.
It adds the project root to the Python path so that modules can be imported correctly.
"""
import sys
import os

# Add project root to Python path
# so that "from decision.state import ..." works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))