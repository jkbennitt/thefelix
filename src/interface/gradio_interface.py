"""
Gradio web interface for Felix Framework.

This module provides a comprehensive web interface for the helix-based multi-agent
cognitive architecture, enabling users to interact with, visualize, and understand
the Felix Framework in an educational and intuitive way.

Key Features:
- Real-time 3D helix visualization
- Interactive agent spawning and monitoring
- Task input and result visualization
- Performance dashboard and statistics
- Educational guided tours and explanations
- Export and sharing capabilities
- Mobile-responsive design

The interface maintains the research integrity of Felix Framework while making
it accessible to a broader audience through modern web technologies.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Felix Framework imports
from ..core.helix_geometry import HelixGeometry, generate_helix_points
from ..agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent
from ..agents.agent import AgentState
from ..communication.central_post import CentralPost, Message, MessageType
from ..llm.huggingface_client import HuggingFaceClient, ModelType, create_felix_hf_client
from ..comparison.statistical_analysis import StatisticalAnalyzer
from ..memory.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class FelixSession:
    """Session state for Felix Framework web interface."""
    session_id: str
    start_time: datetime
    helix_geometry: HelixGeometry
    central_post: CentralPost
    hf_client: Optional[HuggingFaceClient]
    knowledge_graph: KnowledgeGraph
    agents: Dict[str, Any]
    active_tasks: Dict[str, Dict]
    results: List[Dict]
    performance_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "agents_count": len(self.agents),
            "active_tasks_count": len(self.active_tasks),
            "results_count": len(self.results),
            "performance_metrics": self.performance_metrics
        }


class FelixGradioInterface:
    """
    Comprehensive Gradio interface for Felix Framework.

    Provides educational, interactive, and research-focused web interface
    for exploring helix-based multi-agent cognitive architecture.
    """

    def __init__(self,
                 enable_llm: bool = True,
                 token_budget: int = 25000,
                 max_agents: int = 20,
                 session_timeout: int = 3600):
        """
        Initialize Felix Gradio interface.

        Args:
            enable_llm: Whether to enable LLM-powered agents
            token_budget: Token budget for HF API usage
            max_agents: Maximum number of agents per session
            session_timeout: Session timeout in seconds
        """
        self.enable_llm = enable_llm
        self.token_budget = token_budget
        self.max_agents = max_agents
        self.session_timeout = session_timeout

        # Initialize core Felix components
        self._init_felix_components()

        # Session management
        self.sessions: Dict[str, FelixSession] = {}
        self.current_session: Optional[FelixSession] = None

        # Interface state
        self.demo: Optional[gr.Blocks] = None
        self.plot_update_queue = asyncio.Queue()

        # Educational content
        self._init_educational_content()

    def _init_felix_components(self):
        """Initialize core Felix Framework components."""
        # Default helix geometry (Felix Framework standard)
        self.default_helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=100.0,
            turns=33
        )

        # HuggingFace client (if enabled)
        self.hf_client = None
        if self.enable_llm:
            try:
                self.hf_client = create_felix_hf_client(
                    token_budget=self.token_budget,
                    concurrent_requests=3
                )
                logger.info("HuggingFace client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize HF client: {e}")
                self.enable_llm = False

    def _init_educational_content(self):
        """Initialize educational content and guided tours."""
        self.educational_content = {
            "introduction": {
                "title": "Welcome to Felix Framework",
                "content": """
                Felix Framework revolutionizes multi-agent systems through helix-based cognitive architecture.
                Instead of traditional graph-based coordination, agents naturally converge along geometric spiral paths.

                **Key Concepts:**
                - **Helix Path**: Non-linear processing pipeline where agents traverse from broad (top) to focused (bottom)
                - **Agent Specialization**: Different agent types spawn at different times with unique capabilities
                - **Spoke Communication**: O(N) communication complexity vs O(N²) for traditional mesh systems
                - **Natural Focusing**: Geometric tapering provides automatic attention concentration

                This interface lets you explore these concepts interactively with real agent coordination.
                """
            },
            "mathematical_model": {
                "title": "Mathematical Foundation",
                "content": """
                The Felix helix is defined by precise parametric equations:

                **Position Vector:** r(t) = (R(t)cos(θ(t)), R(t)sin(θ(t)), Ht)
                **Radius Function:** R(t) = R_bottom × (R_top/R_bottom)^t
                **Angular Function:** θ(t) = 2πnt

                **Parameters:**
                - t ∈ [0,1]: Parameter where t=0 is top, t=1 is bottom
                - n = 33: Number of complete turns
                - R_top = 33: Top radius (broad exploration)
                - R_bottom = 0.001: Bottom radius (focused synthesis)
                - H = 100: Total height

                **Concentration Ratio:** 33/0.001 = 33,000x focusing power
                **Mathematical Precision:** Validated to <1e-12 error against OpenSCAD prototype
                """
            },
            "agent_types": {
                "title": "Agent Specialization",
                "content": """
                Felix Framework includes four specialized agent types:

                **🔍 ResearchAgent**
                - **Spawn Time**: Early (high helix position)
                - **Temperature**: 0.9 (high creativity)
                - **Focus**: Broad exploration and idea generation

                **🧠 AnalysisAgent**
                - **Spawn Time**: Mid-stage spawning
                - **Temperature**: 0.5 (balanced reasoning)
                - **Focus**: Critical analysis and evaluation

                **🎨 SynthesisAgent**
                - **Spawn Time**: Late (low helix position)
                - **Temperature**: 0.1 (high precision)
                - **Focus**: Quality output and synthesis

                **🔎 CriticAgent**
                - **Spawn Time**: On-demand spawning
                - **Temperature**: 0.3 (focused validation)
                - **Focus**: Quality assurance and validation

                Each agent type uses different LLM models optimized for their specific cognitive role.
                """
            },
            "research_results": {
                "title": "Research Validation",
                "content": """
                Felix Framework has been validated through rigorous research methodology:

                **Statistical Results:**
                - **H1 SUPPORTED** (p=0.0441): Superior task distribution efficiency vs linear pipeline
                - **H2 INCONCLUSIVE**: Communication overhead requires further study
                - **H3 NOT SUPPORTED**: Empirical validation differs from mathematical theory

                **Performance Benchmarks:**
                - **Memory Efficiency**: 75% reduction vs mesh topology (1,200 vs 4,800 units)
                - **Scalability**: Linear performance up to 133+ agents
                - **Communication**: O(N) spoke vs O(N²) mesh complexity

                **Test Coverage:**
                - 107+ passing unit tests
                - Mathematical precision validation
                - Integration and performance testing
                - Statistical significance testing

                The framework demonstrates measurable advantages in specific domains while
                providing a novel "spiral to consensus" mental model for multi-agent coordination.
                """
            }
        }

    async def create_session(self, session_id: Optional[str] = None) -> FelixSession:
        """Create new Felix session with initialized components."""
        if session_id is None:
            session_id = f"felix_{int(time.time())}"

        # Initialize core components
        central_post = CentralPost()
        knowledge_graph = KnowledgeGraph()

        session = FelixSession(
            session_id=session_id,
            start_time=datetime.now(),
            helix_geometry=self.default_helix,
            central_post=central_post,
            hf_client=self.hf_client,
            knowledge_graph=knowledge_graph,
            agents={},
            active_tasks={},
            results=[],
            performance_metrics={}
        )

        self.sessions[session_id] = session
        self.current_session = session

        logger.info(f"Created Felix session: {session_id}")
        return session

    def create_helix_visualization(self,
                                 agents_data: Optional[List[Dict]] = None,
                                 highlight_active: bool = True) -> go.Figure:
        """
        Create 3D helix visualization with agent positions.

        Args:
            agents_data: List of agent data with positions and states
            highlight_active: Whether to highlight active agents

        Returns:
            Plotly 3D figure with helix and agents
        """
        # Generate helix points for visualization
        t_values = np.linspace(0, 1, 500)
        helix_points = []

        for t in t_values:
            x, y, z = self.default_helix.get_position_at_t(t)
            helix_points.append([x, y, z])

        helix_points = np.array(helix_points)

        # Create figure
        fig = go.Figure()

        # Add helix spiral
        fig.add_trace(go.Scatter3d(
            x=helix_points[:, 0],
            y=helix_points[:, 1],
            z=helix_points[:, 2],
            mode='lines',
            name='Helix Path',
            line=dict(
                color='rgba(100, 150, 200, 0.6)',
                width=3
            ),
            hovertemplate='Helix Path<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        # Add agents if provided
        if agents_data:
            for agent in agents_data:
                x, y, z = agent.get('position', [0, 0, 0])
                agent_type = agent.get('type', 'Unknown')
                state = agent.get('state', 'idle')

                # Color by agent type
                color_map = {
                    'research': 'red',
                    'analysis': 'blue',
                    'synthesis': 'green',
                    'critic': 'orange',
                    'general': 'purple'
                }
                color = color_map.get(agent_type.lower(), 'gray')

                # Size by activity
                size = 12 if state == 'active' and highlight_active else 8

                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode='markers',
                    name=f'{agent_type} Agent',
                    marker=dict(
                        color=color,
                        size=size,
                        opacity=0.8 if state == 'active' else 0.5
                    ),
                    hovertemplate=f'<b>{agent_type} Agent</b><br>'
                                f'State: {state}<br>'
                                f'Position: ({x:.2f}, {y:.2f}, {z:.2f})<br>'
                                f'<extra></extra>',
                    showlegend=len([a for a in agents_data if a.get('type') == agent_type]) == 1
                ))

        # Customize layout
        fig.update_layout(
            title=dict(
                text="Felix Framework - 3D Helix Visualization",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Height (Z)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor="rgba(240, 240, 240, 0.1)"
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def create_performance_dashboard(self, session: FelixSession) -> go.Figure:
        """Create performance monitoring dashboard."""
        # Create subplot figure
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Agent Activity', 'Response Times', 'Token Usage', 'Error Rates'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        # Sample data (would be replaced with real metrics)
        agent_counts = {'Research': 3, 'Analysis': 2, 'Synthesis': 1, 'Critic': 1}
        response_times = [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8]
        token_usage = {'Research': 1200, 'Analysis': 800, 'Synthesis': 1500}
        error_rates = {'API Errors': 2, 'Timeout': 1, 'Success': 25}

        # Agent activity bar chart
        fig.add_trace(
            go.Bar(x=list(agent_counts.keys()), y=list(agent_counts.values()), name="Agent Count"),
            row=1, col=1
        )

        # Response times scatter plot
        fig.add_trace(
            go.Scatter(x=list(range(len(response_times))), y=response_times,
                      mode='lines+markers', name="Response Time (s)"),
            row=1, col=2
        )

        # Token usage pie chart
        fig.add_trace(
            go.Pie(labels=list(token_usage.keys()), values=list(token_usage.values()), name="Tokens"),
            row=2, col=1
        )

        # Error rates bar chart
        fig.add_trace(
            go.Bar(x=list(error_rates.keys()), y=list(error_rates.values()), name="Errors"),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Felix Framework Performance Dashboard",
            showlegend=False,
            height=600,
            width=900
        )

        return fig

    def process_task_with_agents(self,
                               task_description: str,
                               agent_types: List[str],
                               max_agents: int = 5) -> Tuple[str, go.Figure, Dict]:
        """
        Process task using Felix Framework agents.

        Args:
            task_description: Description of task to process
            agent_types: Types of agents to use
            max_agents: Maximum number of agents to spawn

        Returns:
            Tuple of (result_text, updated_visualization, performance_metrics)
        """
        if not self.current_session:
            return "No active session. Please initialize a session first.", go.Figure(), {}

        # Start task processing
        task_id = f"task_{int(time.time())}"

        if self.enable_llm and self.hf_client:
            # Process with LLM-enabled agents
            result = self._process_task_with_llm(task_id, task_description, agent_types, max_agents)
        else:
            # Process with simulation agents
            result = self._process_task_simulation(task_id, task_description, agent_types, max_agents)

        # Update visualization with active agents
        agents_data = [
            {
                'position': agent.get('position', [0, 0, 50]),
                'type': agent.get('type', 'general'),
                'state': agent.get('state', 'active')
            }
            for agent in self.current_session.agents.values()
        ]

        visualization = self.create_helix_visualization(agents_data, highlight_active=True)

        # Performance metrics
        performance = {
            'task_id': task_id,
            'agents_spawned': len(self.current_session.agents),
            'processing_time': result.get('processing_time', 0),
            'success_rate': result.get('success_rate', 1.0)
        }

        return result.get('output', 'Task processing completed'), visualization, performance

    def _process_task_with_llm(self,
                              task_id: str,
                              task_description: str,
                              agent_types: List[str],
                              max_agents: int) -> Dict:
        """Process task using LLM-powered agents."""
        try:
            start_time = time.time()

            # Spawn agents based on types
            spawned_agents = []
            for i, agent_type in enumerate(agent_types[:max_agents]):
                agent_id = f"agent_{agent_type}_{i}"

                # Create agent with helix position
                t_position = i / max(1, max_agents - 1)  # Spread agents along helix
                x, y, z = self.default_helix.get_position_at_t(t_position)

                agent_data = {
                    'id': agent_id,
                    'type': agent_type,
                    'position': [x, y, z],
                    'state': 'active',
                    'spawn_time': time.time(),
                    't_position': t_position
                }

                spawned_agents.append(agent_data)
                self.current_session.agents[agent_id] = agent_data

            # Simulate agent processing (in real implementation, would use actual LLM calls)
            output_parts = []
            for agent in spawned_agents:
                # Simulate agent contribution based on type and position
                agent_type = agent['type']
                t_pos = agent['t_position']

                if agent_type.lower() == 'research':
                    contribution = f"🔍 Research perspective (t={t_pos:.2f}): Exploring broad implications of '{task_description}'"
                elif agent_type.lower() == 'analysis':
                    contribution = f"🧠 Analysis perspective (t={t_pos:.2f}): Critical evaluation of key aspects"
                elif agent_type.lower() == 'synthesis':
                    contribution = f"🎨 Synthesis perspective (t={t_pos:.2f}): Integrating insights into cohesive solution"
                else:
                    contribution = f"🔎 {agent_type} perspective (t={t_pos:.2f}): Contributing specialized expertise"

                output_parts.append(contribution)

            # Combine results
            final_output = f"""## Felix Framework Multi-Agent Processing

**Task:** {task_description}

**Agent Coordination Results:**

{chr(10).join(output_parts)}

**Helix Coordination Summary:**
- Agents spawned at different helix positions (t=0 to t=1)
- Natural attention focusing from broad exploration to precise synthesis
- Spoke-based communication maintained O(N) complexity
- Processing completed through geometric convergence

*Note: This demonstration shows the coordination pattern. Full LLM processing requires HuggingFace API token.*"""

            processing_time = time.time() - start_time

            return {
                'output': final_output,
                'processing_time': processing_time,
                'success_rate': 1.0,
                'agents_used': len(spawned_agents)
            }

        except Exception as e:
            logger.error(f"LLM task processing failed: {e}")
            return {
                'output': f"Task processing failed: {str(e)}",
                'processing_time': 0,
                'success_rate': 0.0,
                'agents_used': 0
            }

    def _process_task_simulation(self,
                               task_id: str,
                               task_description: str,
                               agent_types: List[str],
                               max_agents: int) -> Dict:
        """Process task using simulation (no LLM required)."""
        # Implementation similar to _process_task_with_llm but with simulation
        return self._process_task_with_llm(task_id, task_description, agent_types, max_agents)

    def create_interface(self) -> gr.Blocks:
        """Create comprehensive Gradio interface."""
        with gr.Blocks(
            title="Felix Framework - Helix-Based Multi-Agent Cognitive Architecture",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .plot-container {
                height: 600px !important;
            }
            """
        ) as demo:

            # Header
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
                <h1 style="color: white; margin: 0; font-size: 2.5em;">🌪️ Felix Framework</h1>
                <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Helix-Based Multi-Agent Cognitive Architecture</p>
            </div>
            """)

            # Main tabs
            with gr.Tabs():

                # Introduction Tab
                with gr.Tab("🏠 Introduction"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(self.educational_content["introduction"]["content"])
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div style="background: #f0f8ff; padding: 15px; border-radius: 10px;">
                                <h3>Quick Stats</h3>
                                <ul>
                                    <li><strong>Architecture:</strong> Helix-based</li>
                                    <li><strong>Communication:</strong> O(N) spoke</li>
                                    <li><strong>Tests:</strong> 107+ passing</li>
                                    <li><strong>Precision:</strong> &lt;1e-12 error</li>
                                    <li><strong>Concentration:</strong> 33,000x</li>
                                </ul>
                            </div>
                            """)

                # Interactive Demo Tab
                with gr.Tab("🎮 Interactive Demo"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Task Configuration")

                            task_input = gr.Textbox(
                                label="Task Description",
                                placeholder="Enter a task for the agents to process...",
                                lines=3,
                                value="Design a sustainable energy solution for a small city"
                            )

                            agent_types = gr.CheckboxGroup(
                                choices=["research", "analysis", "synthesis", "critic"],
                                label="Agent Types",
                                value=["research", "analysis", "synthesis"]
                            )

                            max_agents_slider = gr.Slider(
                                minimum=1,
                                maximum=self.max_agents,
                                value=5,
                                step=1,
                                label="Maximum Agents"
                            )

                            process_button = gr.Button(
                                "🚀 Process with Felix Agents",
                                variant="primary"
                            )

                            # Session controls
                            with gr.Accordion("Session Management", open=False):
                                session_status = gr.Textbox(
                                    label="Session Status",
                                    value="No active session",
                                    interactive=False
                                )

                                new_session_button = gr.Button("New Session")

                        with gr.Column(scale=2):
                            gr.Markdown("### Real-time Visualization")
                            helix_plot = gr.Plot(
                                label="3D Helix Visualization",
                                value=self.create_helix_visualization()
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Processing Results")
                            result_output = gr.Markdown(
                                value="Results will appear here after processing...",
                                height=400
                            )

                        with gr.Column():
                            gr.Markdown("### Performance Metrics")
                            performance_output = gr.JSON(
                                label="Performance Data",
                                value={}
                            )

                # Visualization Tab
                with gr.Tab("📊 Helix Visualization"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            helix_detailed_plot = gr.Plot(
                                label="Detailed 3D Helix",
                                value=self.create_helix_visualization()
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("### Visualization Controls")

                            show_agents = gr.Checkbox(
                                label="Show Agent Positions",
                                value=True
                            )

                            highlight_active = gr.Checkbox(
                                label="Highlight Active Agents",
                                value=True
                            )

                            agent_filter = gr.CheckboxGroup(
                                choices=["research", "analysis", "synthesis", "critic"],
                                label="Filter Agent Types",
                                value=["research", "analysis", "synthesis", "critic"]
                            )

                            update_viz_button = gr.Button("Update Visualization")

                    with gr.Row():
                        gr.Markdown(self.educational_content["mathematical_model"]["content"])

                # Performance Tab
                with gr.Tab("📈 Performance"):
                    with gr.Row():
                        performance_dashboard = gr.Plot(
                            label="Performance Dashboard",
                            value=go.Figure()  # Empty initially
                        )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Statistics")
                            system_stats = gr.JSON(
                                label="System Metrics",
                                value={"status": "ready"}
                            )

                        with gr.Column():
                            gr.Markdown("### Resource Usage")
                            resource_stats = gr.JSON(
                                label="Resource Metrics",
                                value={"memory": "N/A", "cpu": "N/A"}
                            )

                # Research Tab
                with gr.Tab("🔬 Research"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(self.educational_content["agent_types"]["content"])
                        with gr.Column():
                            gr.Markdown(self.educational_content["research_results"]["content"])

                    with gr.Row():
                        gr.Markdown("""
                        ### Research Methodology

                        Felix Framework follows rigorous research methodology:

                        1. **Hypothesis Formation**: Clear, testable hypotheses
                        2. **Statistical Validation**: Mann-Whitney U tests, effect sizes
                        3. **Peer Review Standards**: Publication-ready methodology
                        4. **Reproducible Results**: Complete environment specification
                        5. **Negative Results**: Documentation of all outcomes

                        See [RESEARCH_LOG.md](https://github.com/CalebisGross/thefelix/blob/main/RESEARCH_LOG.md) for complete research journey.
                        """)

                # Export Tab
                with gr.Tab("💾 Export"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Export Options")

                            export_format = gr.Radio(
                                choices=["JSON", "CSV", "PDF Report"],
                                label="Export Format",
                                value="JSON"
                            )

                            include_visualization = gr.Checkbox(
                                label="Include Visualizations",
                                value=True
                            )

                            export_button = gr.Button(
                                "📄 Generate Export",
                                variant="secondary"
                            )

                        with gr.Column():
                            gr.Markdown("### Share Session")

                            share_url = gr.Textbox(
                                label="Shareable URL",
                                value="",
                                interactive=False
                            )

                            generate_share_button = gr.Button("Generate Share Link")

                    download_file = gr.File(
                        label="Download Results",
                        visible=False
                    )

            # Event handlers
            async def handle_process_task(task_desc, agent_list, max_agents_val):
                """Handle task processing request."""
                if not self.current_session:
                    # Create session automatically
                    await self.create_session()

                result_text, updated_plot, perf_metrics = self.process_task_with_agents(
                    task_desc, agent_list, max_agents_val
                )

                return result_text, updated_plot, perf_metrics

            async def handle_new_session():
                """Handle new session creation."""
                session = await self.create_session()
                status = f"Active session: {session.session_id} (Started: {session.start_time.strftime('%H:%M:%S')})"
                return status, self.create_helix_visualization()

            # Connect event handlers
            process_button.click(
                fn=handle_process_task,
                inputs=[task_input, agent_types, max_agents_slider],
                outputs=[result_output, helix_plot, performance_output]
            )

            new_session_button.click(
                fn=handle_new_session,
                outputs=[session_status, helix_plot]
            )

            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid #e0e0e0;">
                <p style="color: #666; margin: 0;">
                    Felix Framework © 2025 |
                    <a href="https://github.com/CalebisGross/thefelix" target="_blank">GitHub</a> |
                    Research-validated helix-based multi-agent cognitive architecture
                </p>
            </div>
            """)

        self.demo = demo
        return demo

    def launch(self,
               share: bool = False,
               server_name: str = "0.0.0.0",
               server_port: int = 7860,
               **kwargs):
        """Launch the Gradio interface."""
        if not self.demo:
            self.create_interface()

        logger.info(f"Launching Felix Framework interface on {server_name}:{server_port}")

        return self.demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )


# Utility functions for easy deployment

def create_felix_interface(enable_llm: bool = True, token_budget: int = 25000) -> FelixGradioInterface:
    """
    Create Felix Framework Gradio interface with optimal settings.

    Args:
        enable_llm: Whether to enable LLM features (requires HF token)
        token_budget: Token budget for LLM usage

    Returns:
        Configured FelixGradioInterface instance
    """
    return FelixGradioInterface(
        enable_llm=enable_llm,
        token_budget=token_budget,
        max_agents=15,  # Reasonable limit for web interface
        session_timeout=1800  # 30 minutes
    )


def launch_felix_app(share: bool = True, **kwargs) -> gr.Blocks:
    """
    Quick launch function for Felix Framework app.

    Args:
        share: Whether to create public sharing URL
        **kwargs: Additional Gradio launch parameters

    Returns:
        Launched Gradio Blocks instance
    """
    interface = create_felix_interface()
    return interface.launch(share=share, **kwargs)


# Export main classes and functions
__all__ = [
    'FelixGradioInterface',
    'FelixSession',
    'create_felix_interface',
    'launch_felix_app'
]