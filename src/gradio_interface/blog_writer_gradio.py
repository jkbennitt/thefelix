"""
Gradio-specific blog writer interface optimized for web deployment.

This module provides a clean, efficient interface for the Felix blog writer
specifically designed for Gradio deployment with ZeroGPU support.

Key Features:
- Simplified API for Gradio integration
- Built-in progress tracking
- GPU resource management
- Automatic session cleanup
- Cached helix calculations
"""

import gradio as gr
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, Generator
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gradio_interface.felix_gradio_adapter import FelixGradioAdapter, ComplexityLevel
from gradio_interface.progress_tracker import ProgressTracker
from gradio_interface.gpu_manager import GPUResourceManager


class GradioBlogWriter:
    """
    Optimized blog writer for Gradio deployment.

    Provides a clean interface with automatic resource management,
    progress tracking, and GPU optimization for ZeroGPU.
    """

    def __init__(self,
                 enable_gpu: bool = True,
                 enable_cache: bool = True,
                 max_concurrent_users: int = 10):
        """
        Initialize Gradio blog writer.

        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_cache: Enable result caching
            max_concurrent_users: Maximum concurrent users
        """
        self.enable_gpu = enable_gpu
        self.enable_cache = enable_cache
        self.max_concurrent_users = max_concurrent_users

        # Initialize components
        self.adapter = FelixGradioAdapter(
            enable_cache=enable_cache,
            max_sessions=max_concurrent_users,
            session_timeout=300.0,  # 5 minute timeout for web
            default_complexity=ComplexityLevel.MEDIUM
        )

        # GPU resource manager for ZeroGPU
        if enable_gpu:
            self.gpu_manager = GPUResourceManager()
        else:
            self.gpu_manager = None

        # Progress tracker for UI updates
        self.progress_tracker = ProgressTracker()

    def generate_blog_post(self,
                          topic: str,
                          complexity: str = "medium",
                          enable_visualization: bool = False,
                          progress=gr.Progress()) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a blog post using Felix Framework.

        This is the main Gradio callback function.

        Args:
            topic: Blog post topic
            complexity: Complexity level (demo/simple/medium/complex/research)
            enable_visualization: Whether to return visualization data
            progress: Gradio progress callback

        Returns:
            Tuple of (blog_content, metadata_dict)
        """
        if not topic or len(topic.strip()) < 3:
            return "Please provide a valid topic (at least 3 characters).", {}

        # Clean and validate input
        topic = topic.strip()
        complexity = complexity.lower()

        # Map complexity to valid values
        complexity_map = {
            "demo": "demo",
            "simple": "simple",
            "medium": "medium",
            "complex": "complex",
            "research": "research"
        }
        complexity = complexity_map.get(complexity, "medium")

        # Track progress
        with self.progress_tracker.track_operation("blog_generation") as tracker:

            # Update progress
            progress(0.0, desc="Initializing Felix Framework...")
            tracker.update(5, "Initializing")

            # Acquire GPU resources if available
            if self.gpu_manager:
                with self.gpu_manager.acquire_resources(priority="normal") as gpu_context:
                    progress(0.1, desc="GPU resources acquired")
                    tracker.update(10, "GPU ready")

                    # Run generation with GPU acceleration
                    result = self._generate_with_resources(
                        topic, complexity, progress, tracker, gpu_context
                    )
            else:
                # Run without GPU
                result = self._generate_with_resources(
                    topic, complexity, progress, tracker, None
                )

            # Extract content and metadata
            if result and result.get("success"):
                content = result["final_output"]["content"]
                metadata = {
                    "agents_used": len(result.get("agents_participated", [])),
                    "tokens_used": result["final_output"].get("tokens_used", 0),
                    "confidence": result["final_output"].get("confidence", 0.0),
                    "processing_time": tracker.elapsed_time,
                    "complexity": complexity,
                    "cached": result.get("from_cache", False)
                }

                # Add visualization data if requested
                if enable_visualization:
                    metadata["visualization"] = self._create_visualization_data(result)

                progress(1.0, desc="Complete!")
                return content, metadata
            else:
                error_msg = result.get("error", "Unknown error occurred") if result else "Processing failed"
                return f"Error: {error_msg}", {"error": True}

    def _generate_with_resources(self,
                                topic: str,
                                complexity: str,
                                progress: gr.Progress,
                                tracker: Any,
                                gpu_context: Optional[Any]) -> Dict[str, Any]:
        """Generate blog post with resource management."""

        try:
            # Create session
            session_id = self.adapter.create_session()

            # Process request with progress updates
            result = None
            for status_msg, progress_pct in self.adapter.process_blog_request(
                topic=topic,
                complexity=complexity,
                session_id=session_id,
                use_cache=self.enable_cache
            ):
                # Update both Gradio progress and internal tracker
                progress(progress_pct / 100.0, desc=status_msg)
                tracker.update(progress_pct, status_msg)

                # Store last result
                if isinstance(status_msg, dict):
                    result = status_msg

            # Get final result
            if result is None:
                # Generator completed, get the return value
                generator = self.adapter.process_blog_request(
                    topic=topic,
                    complexity=complexity,
                    session_id=session_id,
                    use_cache=self.enable_cache
                )
                for _ in generator:
                    pass  # Exhaust generator
                result = generator.send(None)  # Get return value

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

        finally:
            # Clean up session
            if 'session_id' in locals():
                self.adapter.cleanup_session(session_id)

    def _create_visualization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization data for the helix process."""

        viz_data = {
            "helix_points": [],
            "agent_positions": [],
            "timeline": []
        }

        # Add agent positions
        for agent_info in result.get("agents_participated", []):
            viz_data["agent_positions"].append({
                "agent_id": agent_info["agent_id"],
                "agent_type": agent_info["agent_type"],
                "spawn_time": agent_info["spawn_time"],
                "confidence": agent_info.get("confidence", 0.5)
            })

        # Add timeline events
        for event in result.get("processing_timeline", []):
            viz_data["timeline"].append({
                "time": event.get("time", 0.0),
                "description": event.get("event", "")
            })

        return viz_data

    def create_gradio_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""

        with gr.Blocks(title="Felix Blog Writer") as interface:
            gr.Markdown("""
            # 🌀 Felix Blog Writer

            Generate blog posts using the Felix Framework's helix-based multi-agent orchestration.
            The system uses geometric coordination to naturally converge multiple AI agents toward
            a coherent final output.
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    topic_input = gr.Textbox(
                        label="Blog Topic",
                        placeholder="Enter your blog topic here...",
                        lines=2
                    )

                    complexity_dropdown = gr.Dropdown(
                        label="Complexity Level",
                        choices=[
                            ("Demo (3 agents, fast)", "demo"),
                            ("Simple (5 agents)", "simple"),
                            ("Medium (8 agents)", "medium"),
                            ("Complex (12 agents)", "complex"),
                            ("Research (20 agents)", "research")
                        ],
                        value="medium"
                    )

                    with gr.Row():
                        generate_btn = gr.Button("Generate Blog Post", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### Options")

                    enable_viz = gr.Checkbox(
                        label="Enable Visualization Data",
                        value=False
                    )

                    gr.Markdown("### Session Stats")
                    stats_display = gr.JSON(label="Current Stats", value={})

            output_text = gr.Textbox(
                label="Generated Blog Post",
                lines=20,
                max_lines=50
            )

            metadata_display = gr.JSON(label="Generation Metadata")

            # Set up event handlers
            generate_btn.click(
                fn=self.generate_blog_post,
                inputs=[topic_input, complexity_dropdown, enable_viz],
                outputs=[output_text, metadata_display]
            )

            clear_btn.click(
                fn=lambda: ("", "", {}),
                outputs=[topic_input, output_text, metadata_display]
            )

            # Auto-update stats
            interface.load(
                fn=lambda: self.adapter.get_session_stats(),
                outputs=stats_display,
                every=5  # Update every 5 seconds
            )

        return interface


def create_app(share: bool = False, server_port: int = 7860) -> gr.Blocks:
    """
    Create and configure the Gradio app.

    Args:
        share: Whether to create a public share link
        server_port: Port to run the server on

    Returns:
        Configured Gradio Blocks interface
    """
    writer = GradioBlogWriter(
        enable_gpu=True,  # Enable GPU if available
        enable_cache=True,  # Enable caching
        max_concurrent_users=10
    )

    return writer.create_gradio_interface()


# For direct execution
if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, server_port=7860)