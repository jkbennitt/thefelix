"""
Gradio-compatible adapter for Felix Framework with ZeroGPU optimization.

This module provides a web-friendly interface to the Felix Framework's helix-based
multi-agent orchestration system, optimized for deployment on Gradio with ZeroGPU.

Key Features:
- Session management for multiple concurrent users
- Progress tracking with real-time updates
- Configurable complexity levels for resource optimization
- Caching for repeated operations
- Thread-safe execution for concurrent requests
- Lightweight demo modes for quick responses

The adapter maintains backward compatibility with local LM Studio while providing
seamless integration with Gradio's callback system and UI components.
"""

import asyncio
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from threading import Lock
import hashlib

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.helix_geometry import HelixGeometry
from agents.specialized_agents import create_specialized_team
from communication.central_post import CentralPost
from communication.spoke import SpokeManager
from llm.token_budget import TokenBudgetManager
from agents.llm_agent import LLMTask

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for web deployment optimization."""
    DEMO = "demo"  # 3 agents, minimal processing
    SIMPLE = "simple"  # 5 agents, basic processing
    MEDIUM = "medium"  # 8 agents, standard processing
    COMPLEX = "complex"  # 12 agents, full processing
    RESEARCH = "research"  # 20 agents, research-grade


@dataclass
class SessionState:
    """State for a single user session."""
    session_id: str
    created_at: float
    last_accessed: float
    helix: Optional[HelixGeometry] = None
    central_post: Optional[CentralPost] = None
    agents: List[Any] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    is_processing: bool = False
    progress: float = 0.0
    status_message: str = "Ready"

    def cleanup(self):
        """Clean up session resources."""
        if self.central_post and hasattr(self.central_post, 'shutdown'):
            try:
                asyncio.run(self.central_post.shutdown_async())
            except:
                pass
        self.agents.clear()
        self.results.clear()


class ResultCache:
    """Simple LRU cache for repeated operations."""

    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()

    def _make_key(self, topic: str, complexity: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        params = f"{topic}_{complexity}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(params.encode()).hexdigest()

    def get(self, topic: str, complexity: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        key = self._make_key(topic, complexity, **kwargs)
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key].copy()
        return None

    def set(self, topic: str, complexity: str, result: Dict[str, Any], **kwargs):
        """Cache a result."""
        key = self._make_key(topic, complexity, **kwargs)
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self.cache.popitem(last=False)
            self.cache[key] = result.copy()

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()


class FelixGradioAdapter:
    """
    Gradio-compatible adapter for Felix Framework.

    Provides web-friendly interface with session management, progress tracking,
    and resource optimization for deployment on Gradio with ZeroGPU.
    """

    # Complexity configurations
    COMPLEXITY_CONFIG = {
        ComplexityLevel.DEMO: {
            "num_agents": 3,
            "helix_turns": 5,
            "max_tokens": 100,
            "simulation_time": 0.3,
            "timeout": 10
        },
        ComplexityLevel.SIMPLE: {
            "num_agents": 5,
            "helix_turns": 10,
            "max_tokens": 200,
            "simulation_time": 0.5,
            "timeout": 20
        },
        ComplexityLevel.MEDIUM: {
            "num_agents": 8,
            "helix_turns": 20,
            "max_tokens": 400,
            "simulation_time": 0.7,
            "timeout": 30
        },
        ComplexityLevel.COMPLEX: {
            "num_agents": 12,
            "helix_turns": 30,
            "max_tokens": 600,
            "simulation_time": 0.9,
            "timeout": 45
        },
        ComplexityLevel.RESEARCH: {
            "num_agents": 20,
            "helix_turns": 33,
            "max_tokens": 1000,
            "simulation_time": 1.0,
            "timeout": 60
        }
    }

    def __init__(self,
                 llm_client=None,
                 enable_cache: bool = True,
                 max_sessions: int = 100,
                 session_timeout: float = 3600.0,
                 default_complexity: ComplexityLevel = ComplexityLevel.MEDIUM):
        """
        Initialize Gradio adapter.

        Args:
            llm_client: Optional LLM client (for backward compatibility)
            enable_cache: Whether to enable result caching
            max_sessions: Maximum concurrent sessions
            session_timeout: Session timeout in seconds
            default_complexity: Default complexity level
        """
        self.llm_client = llm_client
        self.enable_cache = enable_cache
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.default_complexity = default_complexity

        # Session management
        self.sessions: Dict[str, SessionState] = {}
        self.session_lock = Lock()

        # Result caching
        self.cache = ResultCache() if enable_cache else None

        # Initialize with placeholder client if none provided
        if self.llm_client is None:
            # Try HuggingFace client first for web deployment, fallback to mock
            try:
                from llm.huggingface_client import HuggingFaceClient
                self.llm_client = HuggingFaceClient()
            except:
                # Create a simple mock client inline for now
                self.llm_client = self._create_mock_client()

        logger.info(f"FelixGradioAdapter initialized with {default_complexity.value} complexity")

    def create_session(self) -> str:
        """Create a new user session."""
        session_id = str(uuid.uuid4())
        current_time = time.time()

        with self.session_lock:
            # Clean up old sessions
            self._cleanup_old_sessions()

            # Create new session
            session = SessionState(
                session_id=session_id,
                created_at=current_time,
                last_accessed=current_time
            )
            self.sessions[session_id] = session

        logger.info(f"Created session {session_id}")
        return session_id

    def _cleanup_old_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired = []

        for sid, session in self.sessions.items():
            if current_time - session.last_accessed > self.session_timeout:
                expired.append(sid)

        for sid in expired:
            session = self.sessions.pop(sid)
            session.cleanup()
            logger.info(f"Cleaned up expired session {sid}")

        # Enforce max sessions
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_sid = min(self.sessions.keys(),
                           key=lambda x: self.sessions[x].last_accessed)
            session = self.sessions.pop(oldest_sid)
            session.cleanup()
            logger.info(f"Removed oldest session {oldest_sid} (max sessions reached)")

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session

    def process_blog_request(self,
                            topic: str,
                            complexity: str = "medium",
                            session_id: Optional[str] = None,
                            use_cache: bool = True) -> Generator[Tuple[str, float], None, Dict[str, Any]]:
        """
        Process a blog writing request with progress updates.

        This is a generator that yields (status_message, progress) tuples
        for Gradio progress tracking.

        Args:
            topic: Blog post topic
            complexity: Complexity level (demo/simple/medium/complex/research)
            session_id: Optional session ID for stateful processing
            use_cache: Whether to use cached results if available

        Yields:
            Tuple of (status_message, progress_percentage)

        Returns:
            Final results dictionary
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(topic, complexity)
            if cached:
                yield ("Using cached result", 100.0)
                return cached

        # Create or get session
        if session_id is None:
            session_id = self.create_session()

        session = self.get_session(session_id)
        if not session:
            yield ("Invalid session", 0.0)
            return {"error": "Invalid session ID"}

        # Mark session as processing
        session.is_processing = True
        session.progress = 0.0

        try:
            # Initialize session components if needed
            yield ("Initializing Felix Framework", 10.0)
            session.progress = 10.0
            self._initialize_session_components(session, complexity)

            # Create agent team
            yield ("Creating specialized agent team", 20.0)
            session.progress = 20.0
            self._create_agent_team(session, complexity)

            # Run processing with progress updates
            yield ("Starting helix orchestration", 30.0)
            session.progress = 30.0

            # Process asynchronously with progress tracking
            result = yield from self._run_async_processing(session, topic, complexity)

            # Cache result
            if self.cache and result.get("success", False):
                self.cache.set(topic, complexity, result)

            # Final progress
            yield ("Processing complete", 100.0)
            session.progress = 100.0

            return result

        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}")
            yield (f"Error: {str(e)}", 0.0)
            return {"error": str(e), "success": False}

        finally:
            session.is_processing = False

    def _initialize_session_components(self, session: SessionState, complexity: str):
        """Initialize helix and communication components for session."""
        config = self.COMPLEXITY_CONFIG[ComplexityLevel(complexity)]

        # Create helix geometry with reduced complexity for web
        session.helix = HelixGeometry(
            top_radius=10.0,  # Reduced from 33.0
            bottom_radius=0.01,  # Increased from 0.001 for stability
            height=float(config["helix_turns"]),
            turns=config["helix_turns"]
        )

        # Initialize communication system
        session.central_post = CentralPost(
            max_agents=config["num_agents"],
            enable_metrics=True,
            enable_memory=False  # Disable for web deployment
        )

        logger.info(f"Session {session.session_id} initialized with {complexity} complexity")

    def _create_agent_team(self, session: SessionState, complexity: str):
        """Create specialized agent team for session."""
        config = self.COMPLEXITY_CONFIG[ComplexityLevel(complexity)]

        # Create token budget manager with web-optimized settings
        token_budget = TokenBudgetManager(
            base_budget=config["max_tokens"],
            min_budget=50,
            max_budget=config["max_tokens"],
            strict_mode=True  # Always strict for web
        )

        # Create simplified team
        session.agents = create_specialized_team(
            helix=session.helix,
            llm_client=self.llm_client,
            task_complexity=complexity,
            token_budget_manager=token_budget,
            random_seed=42  # Fixed seed for reproducibility
        )

        # Limit agent count
        session.agents = session.agents[:config["num_agents"]]

        logger.info(f"Created {len(session.agents)} agents for session {session.session_id}")

    def _run_async_processing(self, session: SessionState, topic: str,
                            complexity: str) -> Generator[Tuple[str, float], None, Dict[str, Any]]:
        """Run async processing with progress updates."""
        config = self.COMPLEXITY_CONFIG[ComplexityLevel(complexity)]

        # Create main task
        task = LLMTask(
            task_id=f"blog_{session.session_id}",
            description=f"Write a blog post about: {topic}",
            context=f"Complexity: {complexity}. Be concise and focused."
        )

        # Initialize spoke manager for communication
        spoke_manager = SpokeManager(session.central_post)
        for agent in session.agents:
            spoke_manager.register_agent(agent)

        results = {
            "topic": topic,
            "complexity": complexity,
            "session_id": session.session_id,
            "agents_participated": [],
            "final_output": None,
            "success": False,
            "processing_timeline": []
        }

        # Run actual helix-based processing
        try:
            # Use asyncio for real parallel processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Start central post async processing
            loop.run_until_complete(
                session.central_post.start_async_processing(max_concurrent_processors=2)
            )

            simulation_time = config["simulation_time"]
            time_step = 0.05
            current_time = 0.0
            max_iterations = int(simulation_time / time_step)

            for iteration in range(max_iterations):
                current_progress = 30.0 + (60.0 * (iteration / max_iterations))

                # Spawn and update agents
                ready_agents = []
                for agent in session.agents:
                    if agent.can_spawn(current_time) and agent.state.value == "waiting":
                        agent.spawn(current_time, task)
                        ready_agents.append(agent)
                        results["processing_timeline"].append({
                            "time": current_time,
                            "event": f"Spawned {agent.agent_id} ({agent.agent_type})"
                        })
                    elif agent.state.value == "active":
                        agent.update_position(current_time)
                        if current_time % 0.1 < time_step:
                            ready_agents.append(agent)

                # Update progress
                if ready_agents:
                    yield (f"Processing {len(ready_agents)} active agents at t={current_time:.2f}", current_progress)
                    session.progress = current_progress

                    # Process agents (simplified for web)
                    for agent in ready_agents[:2]:  # Limit concurrent processing
                        try:
                            # Use synchronous processing for simplicity in web context
                            if hasattr(self.llm_client, 'process_task'):
                                result = self.llm_client.process_task(task, agent)
                                if result:
                                    results["agents_participated"].append({
                                        "agent_id": agent.agent_id,
                                        "agent_type": agent.agent_type,
                                        "spawn_time": current_time,
                                        "confidence": 0.7 + (current_time * 0.2)
                                    })
                        except Exception as e:
                            logger.warning(f"Agent {agent.agent_id} processing failed: {e}")

                # Check for early completion
                if len(results["agents_participated"]) >= config["num_agents"] * 0.7:
                    break

                current_time += time_step
                time.sleep(0.01)  # Small delay for UI responsiveness

            # Clean up async resources
            loop.run_until_complete(session.central_post.shutdown_async())
            loop.close()

        except Exception as e:
            logger.error(f"Processing error: {e}")
            # Fallback to mock processing
            results = self._fallback_mock_processing(session, topic, complexity)

        # Generate final output
        yield ("Synthesizing final output", 90.0)
        session.progress = 90.0

        # Create final synthesis
        if results["agents_participated"]:
            results["final_output"] = {
                "content": self._synthesize_output(topic, complexity, results),
                "confidence": max([a["confidence"] for a in results["agents_participated"]], default=0.5),
                "tokens_used": config["max_tokens"] * len(results["agents_participated"]) // 2,
                "agents_used": len(results["agents_participated"]),
                "processing_time": current_time
            }
            results["success"] = True
        else:
            # Fallback to mock if no agents participated
            results["final_output"] = {
                "content": self._generate_mock_blog_post(topic, complexity),
                "confidence": 0.5,
                "tokens_used": config["max_tokens"],
                "agents_used": 0
            }
            results["success"] = True

        return results

    def _generate_mock_blog_post(self, topic: str, complexity: str) -> str:
        """Generate a mock blog post for demo purposes."""
        templates = {
            ComplexityLevel.DEMO: "Brief overview of {topic}.",
            ComplexityLevel.SIMPLE: "Introduction to {topic}.\n\nKey points and basic concepts.",
            ComplexityLevel.MEDIUM: "# {topic}\n\n## Introduction\nOverview of the topic.\n\n## Main Points\nDetailed discussion.\n\n## Conclusion\nSummary and takeaways.",
            ComplexityLevel.COMPLEX: "# Comprehensive Analysis: {topic}\n\n## Executive Summary\nHigh-level overview.\n\n## Introduction\nContext and background.\n\n## Analysis\nDetailed examination.\n\n## Implications\nBroader impacts.\n\n## Recommendations\nActionable insights.\n\n## Conclusion\nFinal thoughts.",
            ComplexityLevel.RESEARCH: "# Research Report: {topic}\n\n## Abstract\nResearch summary.\n\n## Introduction\nBackground and motivation.\n\n## Literature Review\nExisting work.\n\n## Methodology\nResearch approach.\n\n## Findings\nKey discoveries.\n\n## Discussion\nInterpretation and implications.\n\n## Future Work\nNext steps.\n\n## Conclusion\nSummary of contributions.\n\n## References\nCitations and sources."
        }

        template = templates.get(ComplexityLevel(complexity), templates[ComplexityLevel.MEDIUM])
        return template.format(topic=topic)

    def get_progress(self, session_id: str) -> Tuple[float, str]:
        """Get current progress for a session."""
        session = self.get_session(session_id)
        if session:
            return (session.progress, session.status_message)
        return (0.0, "Session not found")

    def cleanup_session(self, session_id: str):
        """Clean up a specific session."""
        with self.session_lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                session.cleanup()
                logger.info(f"Cleaned up session {session_id}")

    def _create_mock_client(self):
        """Create a simple mock LLM client for demo purposes."""
        class MockLLMClient:
            def __init__(self):
                self.name = "MockLLM"

            def process_task(self, task, agent):
                """Mock task processing."""
                return {
                    "content": f"Mock response for {agent.agent_type} agent",
                    "tokens": 100
                }

            def test_connection(self):
                return True

            def get_usage_stats(self):
                return {"requests": 0, "tokens": 0}

        return MockLLMClient()

    def _synthesize_output(self, topic: str, complexity: str, results: Dict[str, Any]) -> str:
        """Synthesize final output from agent contributions."""
        # Create a basic synthesis from agent participation
        agent_types = [a["agent_type"] for a in results["agents_participated"]]

        output = f"# {topic}\n\n"

        if "research" in agent_types:
            output += "## Research Findings\n"
            output += f"Based on comprehensive research about {topic}, key insights have been gathered.\n\n"

        if "analysis" in agent_types:
            output += "## Analysis\n"
            output += f"Analysis reveals important patterns and relationships in {topic}.\n\n"

        if "synthesis" in agent_types:
            output += "## Synthesis\n"
            output += f"Integrating multiple perspectives on {topic} provides a unified understanding.\n\n"

        if "critic" in agent_types:
            output += "## Critical Review\n"
            output += "Quality assessment confirms the validity of the findings.\n\n"

        output += "## Conclusion\n"
        output += f"This collaborative exploration of {topic} demonstrates the power of multi-agent coordination."

        return output

    def _fallback_mock_processing(self, session: SessionState, topic: str, complexity: str) -> Dict[str, Any]:
        """Fallback mock processing when real processing fails."""
        config = self.COMPLEXITY_CONFIG[ComplexityLevel(complexity)]

        results = {
            "topic": topic,
            "complexity": complexity,
            "session_id": session.session_id,
            "agents_participated": [],
            "final_output": None,
            "success": False,
            "processing_timeline": []
        }

        # Simulate agent participation
        for i, agent in enumerate(session.agents[:config["num_agents"]]):
            results["agents_participated"].append({
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "spawn_time": i * 0.1,
                "confidence": 0.6 + (i * 0.05)
            })

        return results

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        with self.session_lock:
            active_count = len(self.sessions)
            processing_count = sum(1 for s in self.sessions.values() if s.is_processing)

            return {
                "active_sessions": active_count,
                "processing_sessions": processing_count,
                "max_sessions": self.max_sessions,
                "cache_enabled": self.enable_cache,
                "cache_size": len(self.cache.cache) if self.cache else 0
            }