"""
Interface module for Felix Framework.

This module provides web interface components for the Felix Framework,
including Gradio-based interactive interfaces, 3D visualizations,
and educational content delivery.
"""

from .gradio_interface import FelixGradioInterface, FelixSession, create_felix_interface, launch_felix_app

__all__ = [
    'FelixGradioInterface',
    'FelixSession',
    'create_felix_interface',
    'launch_felix_app'
]