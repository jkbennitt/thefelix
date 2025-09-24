#!/usr/bin/env python3
"""
Simple Felix Framework validation for HuggingFace deployment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all critical imports."""
    try:
        from core.helix_geometry import HelixGeometry
        from agents.agent import Agent
        from communication.central_post import CentralPost
        from llm.huggingface_client import HuggingFaceClient
        from interface.gradio_interface import FelixGradioInterface
        import numpy as np
        import plotly.graph_objects as go
        import gradio as gr
        print("[PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_mathematical_precision():
    """Test mathematical precision."""
    try:
        from core.helix_geometry import HelixGeometry
        helix = HelixGeometry(33.0, 0.001, 100.0, 33)
        x, y, z = helix.get_position(0.5)
        print(f"[PASS] Math precision: Position at t=0.5: ({x:.6f}, {y:.6f}, {z:.6f})")
        return True
    except Exception as e:
        print(f"[FAIL] Math precision error: {e}")
        return False

def test_web_interface():
    """Test web interface creation."""
    try:
        from interface.gradio_interface import create_felix_interface
        interface = create_felix_interface(enable_llm=False)
        print("[PASS] Web interface created successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Web interface error: {e}")
        return False

def main():
    print("Felix Framework - Simple Validation")
    print("=" * 40)

    tests = [
        test_imports,
        test_mathematical_precision,
        test_web_interface
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print("=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("SUCCESS: Ready for deployment!")
        return True
    else:
        print("ERROR: Fixes needed before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)