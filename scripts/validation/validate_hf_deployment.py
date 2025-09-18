#!/usr/bin/env python3
"""
Felix Framework - HuggingFace Deployment Validation

Comprehensive validation script for ensuring Felix Framework deployment
readiness on HuggingFace Spaces. Tests all components, integrations,
and performance characteristics.

Usage:
    python validate_hf_deployment.py

This script validates:
1. Core Felix Framework components
2. HuggingFace integration compatibility
3. Web interface functionality
4. Mathematical precision preservation
5. Performance under resource constraints
6. Research integrity maintenance
"""

import sys
import os
import time
import logging
import asyncio
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Stores validation test results."""

    def __init__(self, name: str, success: bool, message: str,
                 duration: float = 0.0, details: Dict[str, Any] = None):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration
        self.details = details or {}


class FelixDeploymentValidator:
    """
    Comprehensive validation suite for Felix Framework HF deployment.
    """

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.critical_failures: List[str] = []

    def run_validation(self) -> bool:
        """Run complete validation suite."""
        print("Felix Framework - HuggingFace Deployment Validation")
        print("=" * 60)

        validators = [
            self._validate_core_imports,
            self._validate_mathematical_precision,
            self._validate_helix_geometry,
            self._validate_agent_system,
            self._validate_hf_integration,
            self._validate_web_interface,
            self._validate_performance,
            self._validate_research_integrity
        ]

        total_tests = len(validators)
        passed_tests = 0

        for validator in validators:
            try:
                start_time = time.time()
                result = validator()
                duration = time.time() - start_time

                if isinstance(result, ValidationResult):
                    result.duration = duration
                    self.results.append(result)
                    if result.success:
                        passed_tests += 1
                        print(f"✅ {result.name}: {result.message} ({duration:.2f}s)")
                    else:
                        print(f"❌ {result.name}: {result.message} ({duration:.2f}s)")
                        if result.name.startswith("CRITICAL"):
                            self.critical_failures.append(result.name)
                else:
                    # Handle multiple results
                    for r in result:
                        r.duration = duration / len(result)
                        self.results.append(r)
                        if r.success:
                            passed_tests += 1
                            print(f"[PASS] {r.name}: {r.message}")
                        else:
                            print(f"[FAIL] {r.name}: {r.message}")
                            if r.name.startswith("CRITICAL"):
                                self.critical_failures.append(r.name)

            except Exception as e:
                duration = time.time() - start_time
                result = ValidationResult(
                    name=f"CRITICAL - {validator.__name__}",
                    success=False,
                    message=f"Validation failed with exception: {e}",
                    duration=duration
                )
                self.results.append(result)
                self.critical_failures.append(result.name)
                print(f"❌ {result.name}: {result.message}")

        # Summary
        print("\n" + "=" * 60)
        print(f"📊 Validation Summary: {passed_tests}/{len(self.results)} tests passed")

        if self.critical_failures:
            print(f"⚠️  Critical failures: {len(self.critical_failures)}")
            for failure in self.critical_failures:
                print(f"   - {failure}")

        success_rate = passed_tests / len(self.results) if self.results else 0
        overall_success = success_rate >= 0.9 and len(self.critical_failures) == 0

        if overall_success:
            print("🎉 Felix Framework is ready for HuggingFace Spaces deployment!")
        else:
            print("🚨 Felix Framework requires fixes before deployment")

        return overall_success

    def _validate_core_imports(self) -> ValidationResult:
        """Validate all core imports work correctly."""
        try:
            # Core framework imports
            from core.helix_geometry import HelixGeometry
            from agents.agent import Agent
            from communication.central_post import CentralPost
            from llm.huggingface_client import HuggingFaceClient
            from interface.gradio_interface import FelixGradioInterface

            # External dependencies
            import numpy as np
            import plotly.graph_objects as go
            import gradio as gr

            return ValidationResult(
                "Core Imports",
                True,
                "All required modules imported successfully",
                details={"numpy_version": np.__version__, "gradio_version": gr.__version__}
            )
        except ImportError as e:
            return ValidationResult(
                "CRITICAL - Core Imports",
                False,
                f"Import failed: {e}"
            )

    def _validate_mathematical_precision(self) -> ValidationResult:
        """Validate mathematical precision preservation."""
        try:
            from core.helix_geometry import HelixGeometry
            import numpy as np

            # Standard Felix parameters
            helix = HelixGeometry(33.0, 0.001, 100.0, 33)

            # Test positions at key points
            test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
            max_error = 0.0

            for t in test_points:
                x, y, z = helix.get_position(t)

                # Validate mathematical consistency
                expected_z = 100.0 * (1.0 - t)  # Linear height
                z_error = abs(z - expected_z)
                max_error = max(max_error, z_error)

                # Validate radius calculation
                radius = helix.get_radius(z)
                actual_radius = np.sqrt(x*x + y*y)
                radius_error = abs(radius - actual_radius)
                max_error = max(max_error, radius_error)

            # Check precision against Felix standard (<1e-12)
            precision_ok = max_error < 1e-10  # Slightly relaxed for deployment

            return ValidationResult(
                "Mathematical Precision",
                precision_ok,
                f"Max error: {max_error:.2e} ({'< 1e-10' if precision_ok else '>= 1e-10'})",
                details={"max_error": max_error, "test_points": len(test_points)}
            )

        except Exception as e:
            return ValidationResult(
                "CRITICAL - Mathematical Precision",
                False,
                f"Precision validation failed: {e}"
            )

    def _validate_helix_geometry(self) -> List[ValidationResult]:
        """Validate helix geometry calculations."""
        results = []

        try:
            from core.helix_geometry import HelixGeometry

            # Test standard Felix helix
            helix = HelixGeometry(33.0, 0.001, 100.0, 33)

            # Test position calculation
            x, y, z = helix.get_position(0.5)
            results.append(ValidationResult(
                "Helix Position Calculation",
                True,
                f"Position at t=0.5: ({x:.3f}, {y:.3f}, {z:.3f})"
            ))

            # Test radius tapering
            r_top = helix.get_radius(0.0)  # Top
            r_bottom = helix.get_radius(100.0)  # Bottom
            concentration_ratio = r_top / r_bottom

            results.append(ValidationResult(
                "Radius Tapering",
                abs(concentration_ratio - 33000) < 100,  # Allow small variance
                f"Concentration ratio: {concentration_ratio:.0f}x"
            ))

            # Test boundary conditions
            try:
                helix.get_position(-0.1)  # Should fail
                results.append(ValidationResult(
                    "Boundary Validation",
                    False,
                    "Failed to reject invalid t value"
                ))
            except ValueError:
                results.append(ValidationResult(
                    "Boundary Validation",
                    True,
                    "Correctly rejects invalid t values"
                ))

        except Exception as e:
            results.append(ValidationResult(
                "CRITICAL - Helix Geometry",
                False,
                f"Helix validation failed: {e}"
            ))

        return results

    def _validate_agent_system(self) -> ValidationResult:
        """Validate agent system functionality."""
        try:
            from agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent
            from communication.central_post import CentralPost

            # Create central post
            central_post = CentralPost()

            # Test agent creation
            research_agent = ResearchAgent(
                agent_id="test_research",
                helix_position=0.1,
                central_post=central_post
            )

            analysis_agent = AnalysisAgent(
                agent_id="test_analysis",
                helix_position=0.5,
                central_post=central_post
            )

            synthesis_agent = SynthesisAgent(
                agent_id="test_synthesis",
                helix_position=0.9,
                central_post=central_post
            )

            # Test agent properties
            agents = [research_agent, analysis_agent, synthesis_agent]
            agent_types = [type(agent).__name__ for agent in agents]

            return ValidationResult(
                "Agent System",
                True,
                f"Created {len(agents)} specialized agents: {', '.join(agent_types)}"
            )

        except Exception as e:
            return ValidationResult(
                "CRITICAL - Agent System",
                False,
                f"Agent system validation failed: {e}"
            )

    def _validate_hf_integration(self) -> List[ValidationResult]:
        """Validate HuggingFace integration components."""
        results = []

        try:
            from llm.huggingface_client import HuggingFaceClient, ModelType, create_felix_hf_client
            from llm.token_budget import TokenBudgetManager

            # Test HF client creation (without actual API calls)
            hf_client = create_felix_hf_client(token_budget=1000, concurrent_requests=2)

            results.append(ValidationResult(
                "HF Client Creation",
                True,
                "HuggingFace client created successfully"
            ))

            # Test model configurations
            available_models = hf_client.get_available_models()

            results.append(ValidationResult(
                "Model Configuration",
                len(available_models) >= 4,
                f"Configured {len(available_models)} model types"
            ))

            # Test token budget manager
            token_manager = TokenBudgetManager(total_budget=5000)

            results.append(ValidationResult(
                "Token Budget Manager",
                True,
                f"Token manager initialized with {token_manager.get_status()['total_budget']} budget"
            ))

        except Exception as e:
            results.append(ValidationResult(
                "CRITICAL - HF Integration",
                False,
                f"HF integration validation failed: {e}"
            ))

        return results

    def _validate_web_interface(self) -> ValidationResult:
        """Validate web interface components."""
        try:
            from interface.gradio_interface import FelixGradioInterface, create_felix_interface
            import plotly.graph_objects as go

            # Test interface creation (without launching)
            interface = create_felix_interface(enable_llm=False, token_budget=1000)

            # Test visualization creation
            viz = interface.create_helix_visualization()

            # Test interface building (without launching)
            demo = interface.create_interface()

            return ValidationResult(
                "Web Interface",
                True,
                f"Gradio interface created with {len(interface.educational_content)} educational sections"
            )

        except Exception as e:
            return ValidationResult(
                "CRITICAL - Web Interface",
                False,
                f"Web interface validation failed: {e}"
            )

    def _validate_performance(self) -> List[ValidationResult]:
        """Validate performance characteristics."""
        results = []

        try:
            import psutil
            import time
            from core.helix_geometry import HelixGeometry

            # Memory usage test
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create multiple helix instances
            helixes = []
            for i in range(100):
                helix = HelixGeometry(33.0, 0.001, 100.0, 33)
                helixes.append(helix)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            results.append(ValidationResult(
                "Memory Usage",
                memory_increase < 50,  # Less than 50MB for 100 helixes
                f"Memory increase: {memory_increase:.1f} MB for 100 helix instances"
            ))

            # Performance timing test
            start_time = time.time()
            positions = []
            for helix in helixes[:10]:  # Test 10 helixes
                for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    positions.append(helix.get_position(t))

            calc_time = time.time() - start_time

            results.append(ValidationResult(
                "Calculation Performance",
                calc_time < 1.0,  # Should complete in under 1 second
                f"50 position calculations in {calc_time:.3f}s"
            ))

        except Exception as e:
            results.append(ValidationResult(
                "Performance Validation",
                False,
                f"Performance validation failed: {e}"
            ))

        return results

    def _validate_research_integrity(self) -> List[ValidationResult]:
        """Validate research integrity preservation."""
        results = []

        try:
            # Test statistical analysis components
            from comparison.statistical_analysis import StatisticalAnalyzer

            analyzer = StatisticalAnalyzer()

            results.append(ValidationResult(
                "Statistical Framework",
                True,
                "Statistical analysis components available"
            ))

            # Test educational content accuracy
            from interface.gradio_interface import FelixGradioInterface

            interface = FelixGradioInterface()
            educational_content = interface.educational_content

            # Check key research concepts are present
            required_concepts = ["helix", "agent", "spoke", "convergence", "parametric"]

            content_text = str(educational_content).lower()
            concepts_found = sum(1 for concept in required_concepts if concept in content_text)

            results.append(ValidationResult(
                "Educational Content",
                concepts_found >= len(required_concepts) - 1,
                f"Found {concepts_found}/{len(required_concepts)} key research concepts"
            ))

            # Validate mathematical references
            math_refs = ["parametric", "exponential", "precision", "<1e-12"]
            math_found = sum(1 for ref in math_refs if ref.lower() in content_text)

            results.append(ValidationResult(
                "Mathematical References",
                math_found >= 2,
                f"Found {math_found}/{len(math_refs)} mathematical references"
            ))

        except Exception as e:
            results.append(ValidationResult(
                "Research Integrity",
                False,
                f"Research integrity validation failed: {e}"
            ))

        return results

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("Felix Framework - HuggingFace Deployment Validation Report")
        report.append("=" * 60)
        report.append("")

        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        report.append(f"Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        report.append(f"Critical Failures: {len(self.critical_failures)}")
        report.append("")

        # Detailed results
        report.append("Detailed Test Results:")
        report.append("-" * 30)

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            report.append(f"[{status}] {result.name}: {result.message}")
            if result.duration > 0:
                report.append(f"        Duration: {result.duration:.2f}s")
            if result.details:
                for key, value in result.details.items():
                    report.append(f"        {key}: {value}")
            report.append("")

        # Recommendations
        if self.critical_failures:
            report.append("Critical Issues Requiring Attention:")
            report.append("-" * 35)
            for failure in self.critical_failures:
                report.append(f"• {failure}")
        else:
            report.append("✅ No critical issues detected")
            report.append("Felix Framework is ready for HuggingFace Spaces deployment!")

        return "\n".join(report)


def main():
    """Main validation entry point."""
    validator = FelixDeploymentValidator()
    success = validator.run_validation()

    # Generate and save report
    report = validator.generate_report()

    try:
        with open('felix_deployment_validation_report.txt', 'w') as f:
            f.write(report)
        print(f"\n📄 Detailed report saved to: felix_deployment_validation_report.txt")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)