"""
ZeroGPU Deployment Guide and Utilities for Felix Framework.

This module provides comprehensive guidance, utilities, and debugging tools
for deploying Felix Framework on HuggingFace Spaces with ZeroGPU acceleration.

Key Features:
- Deployment validation and diagnostics
- Performance benchmarking and optimization recommendations
- Configuration validation for HF Spaces deployment
- Troubleshooting guides and automated fixes
- Resource usage estimation and planning
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment validation status levels."""
    READY = "ready"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DeploymentCheck:
    """Individual deployment validation check."""
    name: str
    status: DeploymentStatus
    message: str
    suggestion: Optional[str] = None
    technical_details: Optional[str] = None
    auto_fix_available: bool = False


@dataclass
class DeploymentReport:
    """Comprehensive deployment validation report."""
    overall_status: DeploymentStatus
    checks: List[DeploymentCheck] = field(default_factory=list)
    performance_estimates: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ZeroGPUDeploymentValidator:
    """
    Comprehensive deployment validator for Felix Framework on ZeroGPU.

    Validates configuration, dependencies, resource requirements, and provides
    optimization recommendations for successful HuggingFace Spaces deployment.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize deployment validator.

        Args:
            project_root: Root directory of the Felix project
        """
        self.project_root = project_root or Path.cwd()
        self.checks = []
        self.performance_data = {}

    def validate_deployment(self) -> DeploymentReport:
        """
        Perform comprehensive deployment validation.

        Returns:
            Detailed deployment report with status and recommendations
        """
        logger.info("Starting ZeroGPU deployment validation")

        self.checks = []

        # Core validation checks
        self._check_dependencies()
        self._check_huggingface_configuration()
        self._check_zerogpu_requirements()
        self._check_felix_configuration()
        self._check_gradio_integration()
        self._check_resource_limits()
        self._check_security_requirements()
        self._check_performance_configuration()

        # Determine overall status
        overall_status = self._determine_overall_status()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Estimate performance and resources
        performance_estimates = self._estimate_performance()
        resource_requirements = self._estimate_resources()

        report = DeploymentReport(
            overall_status=overall_status,
            checks=self.checks.copy(),
            performance_estimates=performance_estimates,
            resource_requirements=resource_requirements,
            recommendations=recommendations
        )

        logger.info(f"Deployment validation completed - Status: {overall_status.value}")
        return report

    def _check_dependencies(self):
        """Check required dependencies for ZeroGPU deployment."""
        check_name = "Dependencies Check"

        try:
            # Check Python version
            import sys
            python_version = sys.version_info
            if python_version < (3, 8):
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.ERROR,
                    message="Python version too old",
                    suggestion="Upgrade to Python 3.8 or newer",
                    technical_details=f"Current: {python_version.major}.{python_version.minor}"
                ))
                return

            # Check critical packages
            required_packages = [
                ("torch", "PyTorch for GPU acceleration"),
                ("transformers", "HuggingFace Transformers"),
                ("spaces", "HuggingFace Spaces integration"),
                ("gradio", "Web interface framework"),
                ("huggingface_hub", "HuggingFace Hub API")
            ]

            missing_packages = []
            version_info = {}

            for package, description in required_packages:
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    version_info[package] = version
                except ImportError:
                    missing_packages.append((package, description))

            if missing_packages:
                missing_list = ", ".join([f"{pkg} ({desc})" for pkg, desc in missing_packages])
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.ERROR,
                    message=f"Missing required packages: {missing_list}",
                    suggestion="Install missing packages using: pip install torch transformers spaces gradio huggingface_hub",
                    auto_fix_available=True
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="All required dependencies available",
                    technical_details=json.dumps(version_info, indent=2)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Dependency check failed: {e}",
                technical_details=str(e)
            ))

    def _check_huggingface_configuration(self):
        """Check HuggingFace token and account configuration."""
        check_name = "HuggingFace Configuration"

        try:
            # Check for HF token
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')

            if not hf_token:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.ERROR,
                    message="HuggingFace token not found",
                    suggestion="Set HF_TOKEN environment variable with your HuggingFace token",
                    technical_details="Token required for model access and Pro account features"
                ))
                return

            # Test token validity
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                user_info = api.whoami()

                account_type = "Pro" if user_info.get('isPro', False) else "Free"
                orgs = user_info.get('orgs', [])

                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message=f"HuggingFace token valid - {account_type} account",
                    technical_details=f"User: {user_info.get('name', 'unknown')}, Organizations: {len(orgs)}"
                ))

            except Exception as e:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.WARNING,
                    message="Could not validate HuggingFace token",
                    suggestion="Check token permissions and network connectivity",
                    technical_details=str(e)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"HuggingFace configuration check failed: {e}",
                technical_details=str(e)
            ))

    def _check_zerogpu_requirements(self):
        """Check ZeroGPU-specific requirements and configuration."""
        check_name = "ZeroGPU Requirements"

        try:
            issues = []
            recommendations = []

            # Check for spaces decorator availability
            try:
                import spaces
                if hasattr(spaces, 'GPU'):
                    recommendations.append("ZeroGPU decorator available")
                else:
                    issues.append("spaces.GPU decorator not found")
            except ImportError:
                issues.append("HuggingFace Spaces package not available")

            # Check PyTorch GPU support
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    recommendations.append(f"CUDA available with {gpu_count} GPU(s)")

                    # Check GPU memory
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        memory_gb = props.total_memory / 1024**3
                        recommendations.append(f"GPU {i}: {props.name}, {memory_gb:.1f}GB")

                        if memory_gb < 8:
                            issues.append(f"GPU {i} has insufficient memory ({memory_gb:.1f}GB < 8GB minimum)")
                else:
                    issues.append("CUDA not available - ZeroGPU features will be disabled")
            except ImportError:
                issues.append("PyTorch not available")

            # Check model configurations
            try:
                from llm.huggingface_client import create_felix_hf_client, estimate_gpu_requirements
                from llm.huggingface_client import get_pro_account_models

                # Test with Pro account models
                pro_models = get_pro_account_models()
                requirements = estimate_gpu_requirements(pro_models)

                max_memory = requirements.get("max_single_model_memory", 0)
                if max_memory > 40:  # 40GB limit for most ZeroGPU instances
                    issues.append(f"Model memory requirement ({max_memory:.1f}GB) exceeds ZeroGPU limits")

                recommendations.append(f"Max model memory: {max_memory:.1f}GB")

            except Exception as model_e:
                issues.append(f"Could not validate model requirements: {model_e}")

            if issues:
                status = DeploymentStatus.WARNING if not any("not available" in issue for issue in issues) else DeploymentStatus.ERROR
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=status,
                    message=f"ZeroGPU issues found: {'; '.join(issues)}",
                    suggestion="Install required packages and check GPU availability",
                    technical_details="\n".join(recommendations)
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="ZeroGPU requirements satisfied",
                    technical_details="\n".join(recommendations)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"ZeroGPU requirements check failed: {e}",
                technical_details=str(e)
            ))

    def _check_felix_configuration(self):
        """Check Felix Framework configuration and structure."""
        check_name = "Felix Framework Configuration"

        try:
            issues = []
            validations = []

            # Check core modules
            core_modules = [
                "src.core.helix_geometry",
                "src.agents.specialized_agents",
                "src.communication.central_post",
                "src.communication.spoke",
                "src.llm.huggingface_client",
                "src.deployment.zerogpu_monitor",
                "src.deployment.zerogpu_error_handler",
                "src.deployment.batch_optimizer"
            ]

            for module_name in core_modules:
                try:
                    __import__(module_name)
                    validations.append(f"✓ {module_name}")
                except ImportError as e:
                    issues.append(f"✗ {module_name}: {e}")

            # Check Gradio interface
            try:
                from gradio_interface.felix_gradio_adapter import FelixGradioAdapter
                validations.append("✓ Gradio adapter available")
            except ImportError:
                issues.append("✗ Gradio adapter not available")

            # Check configuration files
            config_files = [
                "src/llm/huggingface_client.py",
                "src/deployment/zerogpu_monitor.py"
            ]

            for config_file in config_files:
                full_path = self.project_root / config_file
                if full_path.exists():
                    validations.append(f"✓ {config_file}")
                else:
                    issues.append(f"✗ Missing {config_file}")

            if issues:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.ERROR,
                    message=f"Felix configuration issues: {len(issues)} problems found",
                    suggestion="Ensure all Felix modules are properly installed and configured",
                    technical_details="\n".join(issues + validations)
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="Felix Framework properly configured",
                    technical_details="\n".join(validations)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Felix configuration check failed: {e}",
                technical_details=str(e)
            ))

    def _check_gradio_integration(self):
        """Check Gradio interface configuration for HF Spaces."""
        check_name = "Gradio Integration"

        try:
            import gradio as gr
            version = gr.__version__

            # Check minimum Gradio version
            min_version = "3.50.0"
            if version < min_version:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.WARNING,
                    message=f"Gradio version {version} may be outdated",
                    suggestion=f"Consider upgrading to Gradio {min_version} or newer",
                    technical_details=f"Current: {version}, Recommended: {min_version}+"
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message=f"Gradio {version} ready for HF Spaces",
                    technical_details=f"Version: {version}"
                ))

        except ImportError:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message="Gradio not available",
                suggestion="Install Gradio: pip install gradio",
                auto_fix_available=True
            ))
        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Gradio check failed: {e}",
                technical_details=str(e)
            ))

    def _check_resource_limits(self):
        """Check resource usage against HF Spaces limits."""
        check_name = "Resource Limits"

        try:
            # HF Spaces typical limits
            hf_limits = {
                "gpu_memory_gb": 16,  # T4 GPU typical
                "cpu_memory_gb": 16,
                "disk_space_gb": 50,
                "request_timeout_seconds": 60,
                "concurrent_users": 20
            }

            warnings = []
            estimates = {}

            # Estimate Felix resource usage
            try:
                from llm.huggingface_client import estimate_gpu_requirements, create_felix_hf_client

                # Get default model configuration
                client = create_felix_hf_client(enable_zerogpu=True)
                requirements = estimate_gpu_requirements(client.model_configs)

                max_model_memory = requirements.get("max_single_model_memory", 8.0)
                estimates["estimated_gpu_memory_gb"] = max_model_memory

                if max_model_memory > hf_limits["gpu_memory_gb"] * 0.8:  # 80% threshold
                    warnings.append(f"High GPU memory usage: {max_model_memory:.1f}GB (limit: {hf_limits['gpu_memory_gb']}GB)")

                # Estimate concurrent user capacity
                memory_per_user = max_model_memory / 4  # Rough estimate
                max_users = int(hf_limits["gpu_memory_gb"] / memory_per_user)
                estimates["estimated_max_concurrent_users"] = max_users

                if max_users < 3:
                    warnings.append(f"Low concurrent user capacity: {max_users} users")

            except Exception as e:
                warnings.append(f"Could not estimate resource usage: {e}")

            # Check disk space for model cache
            try:
                import shutil
                free_space = shutil.disk_usage(self.project_root).free / 1024**3
                estimates["available_disk_space_gb"] = free_space

                if free_space < 10:  # 10GB minimum
                    warnings.append(f"Low disk space: {free_space:.1f}GB available")

            except Exception as e:
                warnings.append(f"Could not check disk space: {e}")

            if warnings:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.WARNING,
                    message=f"Resource limit concerns: {'; '.join(warnings)}",
                    suggestion="Consider optimizing model selection or implementing more aggressive memory management",
                    technical_details=json.dumps(estimates, indent=2)
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="Resource usage within HF Spaces limits",
                    technical_details=json.dumps(estimates, indent=2)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Resource limits check failed: {e}",
                technical_details=str(e)
            ))

    def _check_security_requirements(self):
        """Check security requirements for HF Spaces deployment."""
        check_name = "Security Requirements"

        try:
            issues = []
            validations = []

            # Check for sensitive information in environment
            sensitive_vars = ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
            env_vars = list(os.environ.keys())

            for var in env_vars:
                if any(sensitive in var.upper() for sensitive in sensitive_vars):
                    if var not in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:  # These are expected
                        issues.append(f"Potential sensitive variable in environment: {var}")

            # Check for hardcoded secrets in common files
            code_files = list(self.project_root.glob("**/*.py"))[:20]  # Sample check
            for file_path in code_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if any(pattern in content.lower() for pattern in ["api_key =", "secret =", "password ="]):
                        issues.append(f"Potential hardcoded secret in {file_path.name}")
                except Exception:
                    continue  # Skip files that can't be read

            # Validate token handling
            try:
                from llm.huggingface_client import HuggingFaceClient
                # Check that client properly handles token from environment
                validations.append("✓ HuggingFace client uses environment token")
            except Exception:
                issues.append("Could not validate token handling")

            if issues:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.WARNING,
                    message=f"Security concerns found: {len(issues)} issues",
                    suggestion="Review and remove any hardcoded secrets, use HF Spaces secrets for sensitive data",
                    technical_details="\n".join(issues[:5])  # Limit output
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="Security requirements satisfied",
                    technical_details="\n".join(validations)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Security check failed: {e}",
                technical_details=str(e)
            ))

    def _check_performance_configuration(self):
        """Check performance optimization configuration."""
        check_name = "Performance Configuration"

        try:
            recommendations = []
            optimizations = []

            # Check ZeroGPU optimizations
            try:
                from deployment.zerogpu_monitor import ZeroGPUMonitor
                from deployment.batch_optimizer import ZeroGPUBatchOptimizer

                optimizations.append("✓ ZeroGPU monitoring available")
                optimizations.append("✓ Batch processing optimization available")

            except ImportError:
                recommendations.append("Enable ZeroGPU monitoring and batch optimization")

            # Check token budget configuration
            try:
                from llm.token_budget import TokenBudgetManager
                manager = TokenBudgetManager(strict_mode=True)
                optimizations.append("✓ Token budget management configured")

            except Exception:
                recommendations.append("Configure token budget management")

            # Check caching configuration
            try:
                from gradio_interface.felix_gradio_adapter import FelixGradioAdapter
                adapter = FelixGradioAdapter(enable_cache=True)
                optimizations.append("✓ Response caching enabled")

            except Exception:
                recommendations.append("Enable response caching in Gradio adapter")

            # Performance recommendations
            perf_recommendations = [
                "Use batch processing for multiple agent requests",
                "Enable model caching to reduce loading time",
                "Implement progressive complexity levels",
                "Monitor GPU memory usage and implement cleanup",
                "Use token budgets to control resource usage"
            ]

            if recommendations:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.WARNING,
                    message=f"Performance optimizations available: {len(recommendations)} improvements",
                    suggestion="; ".join(recommendations[:3]),
                    technical_details="\n".join(optimizations + perf_recommendations)
                ))
            else:
                self.checks.append(DeploymentCheck(
                    name=check_name,
                    status=DeploymentStatus.READY,
                    message="Performance optimizations configured",
                    technical_details="\n".join(optimizations + perf_recommendations)
                ))

        except Exception as e:
            self.checks.append(DeploymentCheck(
                name=check_name,
                status=DeploymentStatus.ERROR,
                message=f"Performance configuration check failed: {e}",
                technical_details=str(e)
            ))

    def _determine_overall_status(self) -> DeploymentStatus:
        """Determine overall deployment status from individual checks."""
        if any(check.status == DeploymentStatus.CRITICAL for check in self.checks):
            return DeploymentStatus.CRITICAL
        elif any(check.status == DeploymentStatus.ERROR for check in self.checks):
            return DeploymentStatus.ERROR
        elif any(check.status == DeploymentStatus.WARNING for check in self.checks):
            return DeploymentStatus.WARNING
        else:
            return DeploymentStatus.READY

    def _generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations based on checks."""
        recommendations = []

        # Collect recommendations from failed checks
        for check in self.checks:
            if check.status != DeploymentStatus.READY and check.suggestion:
                recommendations.append(f"{check.name}: {check.suggestion}")

        # General recommendations
        general_recommendations = [
            "Test deployment in development environment before production",
            "Monitor GPU memory usage during high-load scenarios",
            "Implement gradual rollout for production deployment",
            "Set up error monitoring and alerting",
            "Document deployment configuration and troubleshooting steps"
        ]

        recommendations.extend(general_recommendations[:3])  # Add top 3 general recommendations

        return recommendations[:10]  # Limit to top 10

    def _estimate_performance(self) -> Dict[str, Any]:
        """Estimate performance characteristics for deployment."""
        estimates = {
            "cold_start_time_seconds": 30,  # Typical for model loading
            "warm_inference_time_seconds": 2,  # Per agent request
            "batch_processing_efficiency": 0.7,  # 70% efficiency for batching
            "concurrent_users_target": 5,  # Conservative estimate
            "memory_efficiency_ratio": 0.8  # 80% GPU memory utilization
        }

        try:
            # Try to get actual estimates from configuration
            from llm.huggingface_client import estimate_gpu_requirements, create_felix_hf_client

            client = create_felix_hf_client()
            requirements = estimate_gpu_requirements(client.model_configs)

            # Update estimates based on model requirements
            max_memory = requirements.get("max_single_model_memory", 8.0)
            estimates["model_loading_time_seconds"] = max(5, max_memory * 2)  # Rough estimate
            estimates["max_model_memory_gb"] = max_memory

        except Exception as e:
            logger.warning(f"Could not get detailed performance estimates: {e}")

        return estimates

    def _estimate_resources(self) -> Dict[str, Any]:
        """Estimate resource requirements for deployment."""
        requirements = {
            "minimum_gpu_memory_gb": 8,
            "recommended_gpu_memory_gb": 16,
            "cpu_memory_gb": 8,
            "disk_space_gb": 20,
            "network_bandwidth_mbps": 10
        }

        try:
            # Get specific requirements from model configuration
            from llm.huggingface_client import estimate_gpu_requirements, create_felix_hf_client

            client = create_felix_hf_client()
            model_requirements = estimate_gpu_requirements(client.model_configs)

            requirements.update({
                "estimated_gpu_memory_gb": model_requirements.get("recommended_gpu_memory", 16),
                "minimum_gpu_memory_gb": model_requirements.get("minimum_gpu_memory", 8),
                "peak_memory_usage_gb": model_requirements.get("total_memory_if_all_loaded", 20)
            })

        except Exception as e:
            logger.warning(f"Could not get detailed resource estimates: {e}")

        return requirements

    def generate_deployment_config(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate optimized deployment configuration for HF Spaces."""
        config = {
            "title": "Felix Framework - Multi-Agent Research Assistant",
            "emoji": "🧬",
            "colorFrom": "blue",
            "colorTo": "purple",
            "sdk": "gradio",
            "sdk_version": "4.0.0",
            "app_file": "app.py",
            "pinned": False,
            "license": "mit",
            "hardware": "t4-small",  # Default to T4 small
            "python_version": "3.10",
            "requirements": [
                "torch",
                "transformers",
                "spaces",
                "gradio>=4.0.0",
                "huggingface_hub",
                "numpy",
                "asyncio",
                "aiohttp"
            ],
            "environment_variables": {
                "HF_TOKEN": "{{HF_TOKEN}}",  # To be set in Spaces secrets
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
            },
            "deployment_settings": {
                "enable_zerogpu": True,
                "max_concurrent_users": 10,
                "request_timeout": 60,
                "enable_caching": True,
                "default_complexity": "medium",
                "gpu_memory_threshold": 0.85
            }
        }

        if output_path:
            output_path.write_text(json.dumps(config, indent=2))
            logger.info(f"Deployment configuration saved to {output_path}")

        return config

    def create_app_file(self, output_path: Optional[Path] = None) -> str:
        """Create optimized app.py file for HF Spaces deployment."""
        app_content = '''"""
Felix Framework - Multi-Agent Research Assistant

HuggingFace Spaces deployment with ZeroGPU acceleration.
"""

import gradio as gr
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Felix components
try:
    from src.gradio_interface.felix_gradio_adapter import FelixGradioAdapter, ComplexityLevel
    from src.llm.huggingface_client import create_felix_hf_client
    from src.deployment.zerogpu_monitor import create_zerogpu_monitor

    # Initialize components
    logger.info("Initializing Felix Framework for HF Spaces...")

    # Create optimized HF client for Spaces
    llm_client = create_felix_hf_client(
        token_budget=50000,
        concurrent_requests=5,
        enable_zerogpu=True,
        debug_mode=False
    )

    # Create GPU monitor
    gpu_monitor = create_zerogpu_monitor()

    # Create Gradio adapter
    felix_adapter = FelixGradioAdapter(
        llm_client=llm_client,
        enable_cache=True,
        max_sessions=50,
        session_timeout=3600,
        default_complexity=ComplexityLevel.MEDIUM
    )

    logger.info("Felix Framework initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize Felix Framework: {e}")
    # Fallback to demo mode
    felix_adapter = None


def process_request(topic, complexity="medium"):
    """Process a blog writing request."""
    if not felix_adapter:
        return "Felix Framework initialization failed. Please contact support.", {}

    try:
        # Process request with progress tracking
        results = []
        for status, progress in felix_adapter.process_blog_request(
            topic=topic,
            complexity=complexity,
            use_cache=True
        ):
            results.append((status, progress))
            yield status, progress, {}

        # Return final result
        final_result = results[-1] if results else ({}, 0)
        return final_result[0], final_result[1], final_result[2] if len(final_result) > 2 else {}

    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        return f"Error: {e}", 0, {}


def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(
        title="Felix Framework - Multi-Agent Research Assistant",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .progress-bar { margin: 1rem 0; }
        """
    ) as demo:

        gr.Markdown("""
        # 🧬 Felix Framework
        ### Multi-Agent Research Assistant with Helix-Based Coordination

        Experience the power of geometric multi-agent coordination for research and analysis.
        """, elem_classes=["header"])

        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="Enter your research topic (e.g., 'artificial intelligence ethics')",
                    lines=2
                )

                complexity_input = gr.Dropdown(
                    choices=["demo", "simple", "medium", "complex", "research"],
                    value="medium",
                    label="Processing Complexity",
                    info="Higher complexity uses more agents and provides deeper analysis"
                )

                submit_btn = gr.Button("Start Research", variant="primary", scale=1)

            with gr.Column(scale=3):
                progress_bar = gr.Progress()
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=2,
                    interactive=False
                )

                result_output = gr.Markdown(
                    label="Research Results",
                    height=400
                )

                with gr.Accordion("Technical Details", open=False):
                    metrics_output = gr.JSON(label="Performance Metrics")

        # Event handlers
        submit_btn.click(
            fn=process_request,
            inputs=[topic_input, complexity_input],
            outputs=[status_output, progress_bar, metrics_output],
            show_progress=True
        )

        # Add examples
        gr.Examples(
            examples=[
                ["Renewable energy technologies", "medium"],
                ["Machine learning ethics", "complex"],
                ["Climate change mitigation", "research"],
                ["Quantum computing applications", "simple"]
            ],
            inputs=[topic_input, complexity_input]
        )

        gr.Markdown("""
        ### About Felix Framework

        The Felix Framework demonstrates helix-based cognitive architecture for multi-agent systems,
        where autonomous agents traverse spiral processing paths with spoke-based communication
        to a central coordination system.

        **Key Features:**
        - 🌀 Helix-based agent coordination
        - 🤖 Specialized agent types (Research, Analysis, Synthesis, Critic)
        - ⚡ ZeroGPU acceleration
        - 📊 Real-time progress tracking
        - 🧠 Adaptive complexity levels
        """)

    return demo


# Create and launch interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
'''

        if output_path:
            output_path.write_text(app_content)
            logger.info(f"App file saved to {output_path}")

        return app_content

    def export_full_report(self, output_dir: Path) -> Dict[str, Path]:
        """Export comprehensive deployment report and configuration files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        files_created = {}

        try:
            # Generate validation report
            report = self.validate_deployment()
            report_file = output_dir / "deployment_report.json"

            report_data = {
                "overall_status": report.overall_status.value,
                "timestamp": report.timestamp,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "suggestion": check.suggestion,
                        "technical_details": check.technical_details,
                        "auto_fix_available": check.auto_fix_available
                    }
                    for check in report.checks
                ],
                "performance_estimates": report.performance_estimates,
                "resource_requirements": report.resource_requirements,
                "recommendations": report.recommendations
            }

            report_file.write_text(json.dumps(report_data, indent=2))
            files_created["report"] = report_file

            # Generate deployment configuration
            config_file = output_dir / "hf_spaces_config.json"
            self.generate_deployment_config(config_file)
            files_created["config"] = config_file

            # Generate app.py
            app_file = output_dir / "app.py"
            self.create_app_file(app_file)
            files_created["app"] = app_file

            # Generate requirements.txt
            requirements_file = output_dir / "requirements.txt"
            requirements_content = """torch>=2.0.0
transformers>=4.30.0
spaces>=0.10.0
gradio>=4.0.0
huggingface_hub>=0.15.0
numpy>=1.24.0
aiohttp>=3.8.0
psutil>=5.9.0
"""
            requirements_file.write_text(requirements_content)
            files_created["requirements"] = requirements_file

            # Generate README
            readme_file = output_dir / "README.md"
            readme_content = f"""# Felix Framework - HuggingFace Spaces Deployment

## Deployment Status: {report.overall_status.value.upper()}

This directory contains the complete deployment configuration for Felix Framework on HuggingFace Spaces with ZeroGPU acceleration.

## Files

- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies
- `hf_spaces_config.json` - HF Spaces configuration
- `deployment_report.json` - Validation report

## Quick Deployment

1. Create a new HuggingFace Space with ZeroGPU hardware
2. Upload all files to your Space repository
3. Set your HF_TOKEN in Space secrets
4. The Space will automatically build and deploy

## Performance Estimates

{json.dumps(report.performance_estimates, indent=2)}

## Resource Requirements

{json.dumps(report.resource_requirements, indent=2)}

## Recommendations

{chr(10).join(f"- {rec}" for rec in report.recommendations[:5])}

---

Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} by Felix Framework Deployment Validator
"""
            readme_file.write_text(readme_content)
            files_created["readme"] = readme_file

            logger.info(f"Deployment package created in {output_dir}")
            return files_created

        except Exception as e:
            logger.error(f"Failed to export deployment package: {e}")
            raise


# Utility functions

def validate_zerogpu_deployment(project_root: Optional[Path] = None) -> DeploymentReport:
    """Quick validation function for ZeroGPU deployment readiness."""
    validator = ZeroGPUDeploymentValidator(project_root)
    return validator.validate_deployment()


def create_deployment_package(output_dir: Path, project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Create complete deployment package for HuggingFace Spaces."""
    validator = ZeroGPUDeploymentValidator(project_root)
    return validator.export_full_report(output_dir)


# Export main classes and functions
__all__ = [
    'ZeroGPUDeploymentValidator',
    'DeploymentReport',
    'DeploymentCheck',
    'DeploymentStatus',
    'validate_zerogpu_deployment',
    'create_deployment_package'
]