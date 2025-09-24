#!/usr/bin/env python3
"""
Gradio Blog Writer Application using Felix Framework.

This example demonstrates the refactored Felix Framework optimized for
Gradio deployment with ZeroGPU support. It provides a web interface for
generating blog posts using helix-based multi-agent orchestration.

Features:
- Web-based interface with real-time progress tracking
- Multiple complexity levels for different use cases
- Session management for concurrent users
- Result caching for improved performance
- GPU acceleration when available
- Visualization of agent coordination

Usage:
    python examples/gradio_blog_app.py [--share] [--port PORT]

For HuggingFace Spaces deployment:
    - Place in root directory as app.py
    - Ensure requirements.txt includes gradio and felix dependencies
    - Set space hardware to ZeroGPU for GPU acceleration
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr

# Import Felix components
from gradio_interface.blog_writer_gradio import GradioBlogWriter, create_app
from gradio_interface.felix_gradio_adapter import ComplexityLevel


def create_advanced_interface():
    """Create an advanced Gradio interface with all features."""

    # Initialize the blog writer
    writer = GradioBlogWriter(
        enable_gpu=True,
        enable_cache=True,
        max_concurrent_users=20
    )

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .gr-box {
        border-radius: 8px;
    }
    #output-box {
        min-height: 400px;
    }
    """

    with gr.Blocks(title="Felix Blog Writer", css=custom_css) as app:
        # Header
        gr.Markdown("""
        # 🌀 Felix Blog Writer - Advanced Interface

        Generate high-quality blog posts using the Felix Framework's revolutionary
        helix-based multi-agent orchestration system. Watch as multiple AI agents
        spiral toward consensus to create cohesive, well-structured content.

        ## How it Works

        1. **Enter a topic** - Any subject you'd like a blog post about
        2. **Select complexity** - Choose based on desired depth and agent count
        3. **Click Generate** - Watch the real-time progress as agents collaborate
        4. **Review output** - Get your blog post with detailed metadata

        ---
        """)

        # Main interface
        with gr.Row():
            # Input column
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Input")

                topic_input = gr.Textbox(
                    label="Blog Topic",
                    placeholder="e.g., 'The future of renewable energy', 'Introduction to quantum computing', 'Best practices for remote work'",
                    lines=3,
                    info="Enter any topic you'd like to write about"
                )

                with gr.Row():
                    complexity_dropdown = gr.Dropdown(
                        label="Complexity Level",
                        choices=[
                            ("🎯 Demo (3 agents, ~10s)", "demo"),
                            ("⚡ Simple (5 agents, ~20s)", "simple"),
                            ("🎨 Medium (8 agents, ~30s)", "medium"),
                            ("🚀 Complex (12 agents, ~45s)", "complex"),
                            ("🔬 Research (20 agents, ~60s)", "research")
                        ],
                        value="medium",
                        info="Higher complexity = more agents = better quality (but slower)"
                    )

                    enable_viz = gr.Checkbox(
                        label="Visualization",
                        value=False,
                        info="Include agent coordination data"
                    )

                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate Blog Post",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary"
                    )

                # Examples
                gr.Markdown("### 💡 Example Topics")
                gr.Examples(
                    examples=[
                        ["The impact of artificial intelligence on healthcare"],
                        ["Sustainable living: Simple changes for a greener lifestyle"],
                        ["The psychology of habit formation"],
                        ["Introduction to blockchain technology"],
                        ["The art of effective communication in remote teams"],
                        ["Space exploration: Past achievements and future missions"]
                    ],
                    inputs=topic_input
                )

            # Output column
            with gr.Column(scale=3):
                gr.Markdown("### 📄 Generated Content")

                output_text = gr.Textbox(
                    label="Blog Post",
                    lines=20,
                    max_lines=50,
                    elem_id="output-box",
                    show_copy_button=True
                )

                with gr.Accordion("📊 Generation Metadata", open=False):
                    metadata_display = gr.JSON(
                        label="Detailed Metrics",
                        elem_id="metadata-json"
                    )

                with gr.Accordion("🔍 Agent Activity Log", open=False):
                    activity_log = gr.Textbox(
                        label="Agent Timeline",
                        lines=10,
                        interactive=False
                    )

        # System status
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 System Status")

                with gr.Row():
                    session_stats = gr.JSON(
                        label="Current Sessions",
                        value=writer.adapter.get_session_stats()
                    )

                    gpu_stats = gr.JSON(
                        label="GPU Resources",
                        value=writer.gpu_manager.get_statistics() if writer.gpu_manager else {"gpu_enabled": False}
                    )

        # Info tabs
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Felix Framework

            The Felix Framework implements a novel helix-based cognitive architecture for
            multi-agent coordination. Instead of traditional linear pipelines or graph-based
            orchestration, agents traverse a 3D helical path, naturally converging toward
            consensus as they spiral from the wide top to the narrow bottom.

            ### Key Features:
            - **Geometric Coordination**: Agents' positions in 3D space influence behavior
            - **Natural Convergence**: The tapering helix guides agents toward agreement
            - **Parallel Processing**: Multiple agents work simultaneously
            - **Adaptive Complexity**: Scale from 3 to 20+ agents based on needs

            ### Research Foundation:
            - Mathematical precision validated to <1e-12 error
            - Statistical hypothesis testing with p<0.05 significance
            - O(N) communication complexity via spoke architecture
            - 4,119x attention focusing through geometric tapering
            """)

        with gr.Tab("🎯 Best Practices"):
            gr.Markdown("""
            ## Tips for Best Results

            ### Topic Selection:
            - **Be specific**: "Benefits of meditation for stress" > "meditation"
            - **Add context**: Include target audience or perspective when relevant
            - **Avoid ambiguity**: Clear topics produce more focused content

            ### Complexity Guidelines:
            - **Demo**: Quick tests and simple overviews
            - **Simple**: Blog posts, basic explanations
            - **Medium**: In-depth articles, detailed guides
            - **Complex**: Comprehensive analyses, multi-perspective content
            - **Research**: Academic-style papers, exhaustive coverage

            ### Performance Tips:
            - Results are cached - regenerating the same topic is instant
            - GPU acceleration provides 2-3x speedup when available
            - Lower complexity levels are perfect for drafts and ideation
            """)

        # Event handlers
        def generate_with_activity_log(topic, complexity, viz):
            """Generate blog post and format activity log."""
            content, metadata = writer.generate_blog_post(
                topic, complexity, viz, progress=gr.Progress()
            )

            # Format activity log
            log_text = "Agent Activity Timeline:\n" + "="*50 + "\n"

            if metadata and "visualization" in metadata:
                viz_data = metadata["visualization"]
                for event in viz_data.get("timeline", []):
                    log_text += f"[t={event['time']:.2f}] {event['description']}\n"
            elif metadata and "agents_used" in metadata:
                log_text += f"Total agents used: {metadata['agents_used']}\n"
                log_text += f"Processing time: {metadata.get('processing_time', 'N/A')}s\n"

            return content, metadata, log_text

        generate_btn.click(
            fn=generate_with_activity_log,
            inputs=[topic_input, complexity_dropdown, enable_viz],
            outputs=[output_text, metadata_display, activity_log]
        )

        clear_btn.click(
            fn=lambda: ("", "", {}, ""),
            outputs=[topic_input, output_text, metadata_display, activity_log]
        )

        # Auto-refresh stats
        app.load(
            fn=lambda: (
                writer.adapter.get_session_stats(),
                writer.gpu_manager.get_statistics() if writer.gpu_manager else {"gpu_enabled": False}
            ),
            outputs=[session_stats, gpu_stats],
            every=5  # Refresh every 5 seconds
        )

    return app


def create_simple_interface():
    """Create a simple, streamlined Gradio interface."""

    app = create_app(share=False)
    return app


def main():
    """Main entry point for the Gradio application."""

    parser = argparse.ArgumentParser(
        description="Felix Blog Writer Gradio Application"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple interface instead of advanced"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Create interface
    if args.simple:
        print("Loading simple interface...")
        app = create_simple_interface()
    else:
        print("Loading advanced interface...")
        app = create_advanced_interface()

    # Launch configuration
    launch_config = {
        "share": args.share,
        "server_port": args.port,
        "server_name": "0.0.0.0" if args.share else "127.0.0.1",
        "show_error": True,
        "quiet": False
    }

    print(f"\nStarting Felix Blog Writer on port {args.port}")
    print(f"GPU acceleration: {'Enabled' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'Disabled'}")

    if args.share:
        print("Creating public share link...")

    # Launch the app
    app.launch(**launch_config)


if __name__ == "__main__":
    main()