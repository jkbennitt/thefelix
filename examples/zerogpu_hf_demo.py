#!/usr/bin/env python3
"""
ZeroGPU HuggingFace Client Demo for Felix Framework

This example demonstrates the ZeroGPU-optimized HuggingFace client features:
- GPU-accelerated inference with automatic fallback
- Batch processing for multiple agents
- HF Pro account optimizations
- LMStudioClient compatibility
- Felix agent system integration

Requirements:
- HF_TOKEN environment variable set
- HF Pro account (optional, for premium models)
- ZeroGPU environment (when deployed to HF Spaces)

Usage:
    python examples/zerogpu_hf_demo.py --task "AI ethics in healthcare"
    python examples/zerogpu_hf_demo.py --batch --agents 4
    python examples/zerogpu_hf_demo.py --benchmark --zerogpu
"""

import asyncio
import argparse
import time
import json
import os
from typing import List, Dict, Any

from src.llm.huggingface_client import (
    create_felix_hf_client,
    create_default_client,
    get_pro_account_models,
    estimate_gpu_requirements,
    ModelType,
    HFModelConfig
)
from src.agents.agent import Agent
from src.communication.central_post import CentralPost
from src.core.helix_geometry import HelixGeometry


class ZeroGPUDemo:
    """Demonstration of ZeroGPU HuggingFace client capabilities."""

    def __init__(self, enable_zerogpu: bool = True, use_pro_models: bool = False, debug: bool = False):
        """
        Initialize ZeroGPU demo.

        Args:
            enable_zerogpu: Enable ZeroGPU acceleration
            use_pro_models: Use HF Pro account premium models
            debug: Enable debug logging
        """
        self.enable_zerogpu = enable_zerogpu
        self.use_pro_models = use_pro_models
        self.debug = debug

        # Check environment
        if not os.getenv('HF_TOKEN'):
            raise ValueError("HF_TOKEN environment variable required")

        # Initialize client
        if use_pro_models:
            pro_configs = get_pro_account_models()
            self.client = create_felix_hf_client(
                concurrent_requests=6,
                enable_zerogpu=enable_zerogpu,
                debug_mode=debug
            )
            # Override with Pro models
            self.client.model_configs.update(pro_configs)
        else:
            self.client = create_felix_hf_client(
                enable_zerogpu=enable_zerogpu,
                debug_mode=debug
            )

        print(f"🚀 ZeroGPU HF Client initialized:")
        print(f"   ZeroGPU enabled: {self.client.enable_zerogpu}")
        print(f"   Pro models: {use_pro_models}")
        print(f"   Debug mode: {debug}")

        # Display GPU requirements
        requirements = estimate_gpu_requirements(self.client.model_configs)
        print(f"   GPU requirements: {requirements['recommended_gpu_memory']:.1f} GB recommended")

    async def demo_basic_completion(self, task: str = "renewable energy research") -> Dict[str, Any]:
        """
        Demonstrate basic text completion with ZeroGPU optimization.

        Args:
            task: Task description for completion

        Returns:
            Results dictionary with response and metrics
        """
        print(f"\n📝 Basic Completion Demo: {task}")

        start_time = time.time()

        # Test different agent types
        results = {}
        for agent_type in [ModelType.RESEARCH, ModelType.ANALYSIS, ModelType.SYNTHESIS]:
            print(f"   Testing {agent_type.value} agent...")

            response = await self.client.generate_text(
                prompt=f"As a {agent_type.value} specialist, please provide insights on: {task}",
                agent_type=agent_type,
                temperature=0.7,
                max_tokens=200
            )

            results[agent_type.value] = {
                "content": response.content,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time,
                "gpu_time": response.gpu_time,
                "fallback_used": response.fallback_used,
                "success": response.success
            }

            if self.debug:
                method = "ZeroGPU" if not response.fallback_used else "Inference API"
                print(f"      {method}: {response.tokens_used} tokens, {response.response_time:.2f}s")

        total_time = time.time() - start_time

        return {
            "task": task,
            "results": results,
            "total_time": total_time,
            "client_stats": self.client.get_performance_stats()
        }

    async def demo_batch_processing(self, num_agents: int = 4) -> Dict[str, Any]:
        """
        Demonstrate batch processing for multiple agents.

        Args:
            num_agents: Number of agents to simulate

        Returns:
            Batch processing results and metrics
        """
        print(f"\n⚡ Batch Processing Demo: {num_agents} agents")

        tasks = [
            "Analyze climate change impacts",
            "Research renewable energy solutions",
            "Synthesize policy recommendations",
            "Critique current approaches"
        ][:num_agents]

        agent_types = [
            ModelType.ANALYSIS,
            ModelType.RESEARCH,
            ModelType.SYNTHESIS,
            ModelType.CRITIC
        ][:num_agents]

        start_time = time.time()

        # Test ZeroGPU batching
        print("   Using ZeroGPU batching...")
        batch_results = await self.client.batch_generate(
            prompts=[f"Please {task}" for task in tasks],
            agent_types=agent_types,
            use_zerogpu_batching=self.enable_zerogpu,
            temperature=0.6,
            max_tokens=150
        )

        batch_time = time.time() - start_time

        # Test individual processing for comparison
        print("   Using individual processing...")
        individual_start = time.time()

        individual_results = []
        for task, agent_type in zip(tasks, agent_types):
            result = await self.client.generate_text(
                prompt=f"Please {task}",
                agent_type=agent_type,
                use_zerogpu=False,  # Force individual processing
                temperature=0.6,
                max_tokens=150
            )
            individual_results.append(result)

        individual_time = time.time() - individual_start

        # Calculate metrics
        batch_tokens = sum(r.tokens_used for r in batch_results)
        individual_tokens = sum(r.tokens_used for r in individual_results)

        print(f"   Batch processing: {batch_time:.2f}s, {batch_tokens} tokens")
        print(f"   Individual processing: {individual_time:.2f}s, {individual_tokens} tokens")
        print(f"   Speed improvement: {individual_time / batch_time:.2f}x")

        return {
            "num_agents": num_agents,
            "batch_time": batch_time,
            "individual_time": individual_time,
            "speed_improvement": individual_time / batch_time,
            "batch_tokens": batch_tokens,
            "individual_tokens": individual_tokens,
            "batch_results": [
                {
                    "content": r.content,
                    "tokens": r.tokens_used,
                    "response_time": r.response_time,
                    "batch_processed": r.batch_processed
                }
                for r in batch_results
            ]
        }

    async def demo_felix_agent_integration(self, task: str = "sustainable technology") -> Dict[str, Any]:
        """
        Demonstrate integration with Felix agent system.

        Args:
            task: Task for Felix agents to process

        Returns:
            Felix integration results
        """
        print(f"\n🧠 Felix Agent Integration Demo: {task}")

        # Create Felix components
        helix = HelixGeometry()
        central_post = CentralPost()

        # Create agents with different spawn times (Felix pattern)
        agents = []
        agent_configs = [
            ("research_agent", ModelType.RESEARCH, 0.0),
            ("analysis_agent", ModelType.ANALYSIS, 0.2),
            ("synthesis_agent", ModelType.SYNTHESIS, 0.6),
            ("critic_agent", ModelType.CRITIC, 0.8)
        ]

        for agent_id, agent_type, spawn_time in agent_configs:
            # Get position on helix based on spawn time
            position = helix.get_position_at_parameter(spawn_time)

            # Create system prompt using Felix helix positioning
            system_prompt = self.client.create_agent_system_prompt(
                agent_type.value,
                position,
                f"Task: {task}"
            )

            agents.append({
                "id": agent_id,
                "type": agent_type,
                "position": position,
                "system_prompt": system_prompt,
                "spawn_time": spawn_time
            })

        # Process agents in Felix-style progression
        results = {}
        start_time = time.time()

        for agent in agents:
            print(f"   Processing {agent['id']} at depth {agent['position']['depth_ratio']:.2f}")

            # Use LMStudioClient-compatible interface
            try:
                response = await self.client.complete_async(
                    agent_id=agent['id'],
                    system_prompt=agent['system_prompt'],
                    user_prompt=f"Focus on {task} from your specialized perspective.",
                    temperature=0.1 + agent['position']['depth_ratio'] * 0.8,  # Temperature by depth
                    max_tokens=int(200 + agent['position']['depth_ratio'] * 300)  # More tokens at depth
                )

                results[agent['id']] = {
                    "content": response.content,
                    "tokens_used": response.tokens_used,
                    "response_time": response.response_time,
                    "position": agent['position'],
                    "success": True
                }

            except Exception as e:
                print(f"   Error processing {agent['id']}: {e}")
                results[agent['id']] = {
                    "error": str(e),
                    "success": False
                }

        total_time = time.time() - start_time

        return {
            "task": task,
            "agents_processed": len(agents),
            "total_time": total_time,
            "results": results,
            "helix_progression": "top-to-bottom convergence",
            "final_stats": self.client.get_usage_stats()
        }

    async def demo_error_handling(self) -> Dict[str, Any]:
        """
        Demonstrate error handling and fallback mechanisms.

        Returns:
            Error handling demonstration results
        """
        print(f"\n🛡️ Error Handling & Fallback Demo")

        scenarios = []

        # Test connection handling
        print("   Testing connection handling...")
        connection_ok = self.client.test_connection()
        scenarios.append({
            "test": "connection_test",
            "success": connection_ok
        })

        # Test ZeroGPU fallback (if enabled)
        if self.enable_zerogpu:
            print("   Testing ZeroGPU fallback...")
            try:
                # Force ZeroGPU usage with a reasonable prompt
                response = await self.client.generate_text(
                    "Test fallback mechanism",
                    ModelType.GENERAL,
                    use_zerogpu=True,
                    max_tokens=50
                )

                scenarios.append({
                    "test": "zerogpu_operation",
                    "success": response.success,
                    "fallback_used": response.fallback_used,
                    "method": "ZeroGPU" if not response.fallback_used else "Inference API"
                })

            except Exception as e:
                scenarios.append({
                    "test": "zerogpu_operation",
                    "success": False,
                    "error": str(e)
                })

        # Test token budget handling
        print("   Testing token budget handling...")
        try:
            # Request with very small token limit
            response = await self.client.generate_text(
                "Very brief response please",
                ModelType.GENERAL,
                max_tokens=10
            )

            scenarios.append({
                "test": "token_budget",
                "success": response.success,
                "tokens_used": response.tokens_used
            })

        except Exception as e:
            scenarios.append({
                "test": "token_budget",
                "success": False,
                "error": str(e)
            })

        return {
            "scenarios_tested": len(scenarios),
            "scenarios": scenarios,
            "client_resilient": all(s.get("success", False) for s in scenarios if "error" not in s)
        }

    def print_final_report(self, results: List[Dict[str, Any]]):
        """
        Print comprehensive demo report.

        Args:
            results: List of demo results to report
        """
        print("\n" + "="*60)
        print("🎯 ZEROGPU HUGGINGFACE CLIENT DEMO REPORT")
        print("="*60)

        for i, result in enumerate(results):
            print(f"\nDemo {i+1}: {result.get('demo_name', 'Unknown')}")
            print(f"Duration: {result.get('total_time', 0):.2f}s")

            if 'success_rate' in result:
                print(f"Success Rate: {result['success_rate']:.1%}")

            if 'speed_improvement' in result:
                print(f"Speed Improvement: {result['speed_improvement']:.2f}x")

        # Final client stats
        final_stats = self.client.get_performance_stats()
        print(f"\n📊 Final Client Statistics:")
        print(f"Total Requests: {final_stats['total_requests']}")
        print(f"Total Tokens: {final_stats['total_tokens']}")
        print(f"Average Response Time: {final_stats['average_response_time']:.2f}s")
        print(f"Error Rate: {final_stats['error_rate']:.1%}")

        if final_stats.get('zerogpu_enabled'):
            print(f"ZeroGPU Features: ✅ Available")
            if 'gpu_memory_allocated' in final_stats:
                print(f"GPU Memory Used: {final_stats['gpu_memory_allocated']:.2f} GB")
        else:
            print(f"ZeroGPU Features: ❌ Not Available (using Inference API)")

        print("\n" + "="*60)


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="ZeroGPU HuggingFace Client Demo")
    parser.add_argument("--task", default="AI ethics in healthcare",
                       help="Task description for demos")
    parser.add_argument("--batch", action="store_true",
                       help="Run batch processing demo")
    parser.add_argument("--felix", action="store_true",
                       help="Run Felix integration demo")
    parser.add_argument("--agents", type=int, default=4,
                       help="Number of agents for batch demo")
    parser.add_argument("--zerogpu", action="store_true",
                       help="Enable ZeroGPU acceleration")
    parser.add_argument("--pro", action="store_true",
                       help="Use HF Pro account models")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run all demos for benchmarking")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Initialize demo
    try:
        demo = ZeroGPUDemo(
            enable_zerogpu=args.zerogpu,
            use_pro_models=args.pro,
            debug=args.debug
        )
    except ValueError as e:
        print(f"❌ Demo initialization failed: {e}")
        return 1

    results = []

    try:
        # Run basic completion demo
        print("🚀 Starting ZeroGPU HuggingFace Client demonstrations...")

        basic_result = await demo.demo_basic_completion(args.task)
        basic_result['demo_name'] = 'Basic Completion'
        results.append(basic_result)

        # Run batch processing demo
        if args.batch or args.benchmark:
            batch_result = await demo.demo_batch_processing(args.agents)
            batch_result['demo_name'] = 'Batch Processing'
            results.append(batch_result)

        # Run Felix integration demo
        if args.felix or args.benchmark:
            felix_result = await demo.demo_felix_agent_integration(args.task)
            felix_result['demo_name'] = 'Felix Integration'
            results.append(felix_result)

        # Run error handling demo
        if args.benchmark:
            error_result = await demo.demo_error_handling()
            error_result['demo_name'] = 'Error Handling'
            results.append(error_result)

        # Print comprehensive report
        demo.print_final_report(results)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📄 Results saved to {args.output}")

        return 0

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if hasattr(demo, 'client'):
            await demo.client.close_async()


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))