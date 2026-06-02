# tests/conftest.py
import sys
import os

# Projekt-Root zum Python-Pfad hinzufügen
# damit "from decision.state import ..." funktioniert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))