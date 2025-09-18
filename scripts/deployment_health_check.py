#!/usr/bin/env python3
"""
Comprehensive deployment health check and validation for Felix Framework.
Validates all components and provides detailed health status.
"""

import os
import sys
import time
import json
import asyncio
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class HealthCheckResult:
    """Health check result container."""

    def __init__(self, component: str, status: str, message: str,
                 details: Optional[Dict[str, Any]] = None,
                 duration: Optional[float] = None):
        self.component = component
        self.status = status  # 'healthy', 'warning', 'critical', 'unknown'
        self.message = message
        self.details = details or {}
        self.duration = duration
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'duration': self.duration,
            'timestamp': self.timestamp
        }

class DeploymentHealthChecker:
    """Comprehensive health checker for Felix Framework deployment."""

    def __init__(self, deployment_url: Optional[str] = None, timeout: float = 30.0):
        self.deployment_url = deployment_url or "http://localhost:7860"
        self.timeout = timeout
        self.logger = self._setup_logging()
        self.results: List[HealthCheckResult] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for health checks."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        self.logger.info("🔍 Starting comprehensive deployment health check...")

        # Core component checks
        await self._check_system_health()
        await self._check_felix_core()
        await self._check_environment_configuration()
        await self._check_dependencies()

        # Deployment-specific checks
        if self.deployment_url.startswith("http"):
            await self._check_web_service()
            await self._check_api_endpoints()
            await self._check_gradio_interface()

        # Performance and optimization checks
        await self._check_performance_baselines()
        await self._check_gpu_availability()
        await self._check_memory_usage()

        # Security and configuration checks
        await self._check_security_configuration()
        await self._check_secrets_management()

        return self._generate_health_report()

    async def _check_system_health(self):
        """Check basic system health metrics."""
        start_time = time.time()

        try:
            # CPU and memory checks
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            details = {
                'cpu_usage_percent': cpu_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100
            }

            # Determine status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or details['disk_percent'] > 90:
                status = 'critical'
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {details['disk_percent']:.1f}%"
            elif cpu_percent > 70 or memory.percent > 70 or details['disk_percent'] > 70:
                status = 'warning'
                message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {details['disk_percent']:.1f}%"
            else:
                status = 'healthy'
                message = f"System resources normal: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {details['disk_percent']:.1f}%"

            self.results.append(HealthCheckResult(
                component='system_health',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='system_health',
                status='critical',
                message=f"System health check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_felix_core(self):
        """Check Felix Framework core components."""
        start_time = time.time()

        try:
            # Test core helix geometry
            from core.helix_geometry import HelixGeometry

            helix = HelixGeometry(33.0, 0.001, 100.0, 33)

            # Validate mathematical precision
            test_positions = []
            for i in range(100):
                t = i / 99.0
                pos = helix.get_position_at_t(t)
                test_positions.append(pos)

            # Check edge cases
            top_pos = helix.get_position_at_t(0.0)
            bottom_pos = helix.get_position_at_t(1.0)

            # Validate precision
            top_radius = (top_pos[0]**2 + top_pos[1]**2)**0.5
            bottom_radius = (bottom_pos[0]**2 + bottom_pos[1]**2)**0.5

            precision_error = abs(bottom_radius - 0.001)

            details = {
                'helix_turns': 33,
                'top_radius': float(top_radius),
                'bottom_radius': float(bottom_radius),
                'precision_error': float(precision_error),
                'test_positions_computed': len(test_positions),
                'mathematical_precision_valid': precision_error < 1e-10
            }

            if precision_error < 1e-10:
                status = 'healthy'
                message = f"Felix core validated: {len(test_positions)} positions, precision error {precision_error:.2e}"
            else:
                status = 'warning'
                message = f"Felix core precision degraded: error {precision_error:.2e}"

            self.results.append(HealthCheckResult(
                component='felix_core',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='felix_core',
                status='critical',
                message=f"Felix core check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_environment_configuration(self):
        """Check environment configuration and variables."""
        start_time = time.time()

        try:
            env_vars = {
                'HF_TOKEN': os.getenv('HF_TOKEN'),
                'SPACES_ZERO_GPU': os.getenv('SPACES_ZERO_GPU'),
                'FELIX_DEBUG': os.getenv('FELIX_DEBUG'),
                'FELIX_TOKEN_BUDGET': os.getenv('FELIX_TOKEN_BUDGET'),
                'PORT': os.getenv('PORT'),
                'SPACE_ID': os.getenv('SPACE_ID'),
                'ENVIRONMENT': os.getenv('ENVIRONMENT')
            }

            # Check critical configurations
            issues = []
            warnings = []

            if not env_vars['HF_TOKEN']:
                warnings.append("HF_TOKEN not set - LLM features will be disabled")

            if env_vars['SPACES_ZERO_GPU'] != 'true':
                warnings.append("ZeroGPU not enabled - performance may be limited")

            if not env_vars['PORT']:
                issues.append("PORT not configured")

            # Python version check
            python_version = sys.version_info
            if python_version < (3, 11):
                issues.append(f"Python version {python_version.major}.{python_version.minor} < 3.11")

            details = {
                'environment_variables': {k: bool(v) if k == 'HF_TOKEN' else v for k, v in env_vars.items()},
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'platform': sys.platform,
                'issues': issues,
                'warnings': warnings
            }

            if issues:
                status = 'critical'
                message = f"Environment configuration issues: {', '.join(issues)}"
            elif warnings:
                status = 'warning'
                message = f"Environment warnings: {', '.join(warnings)}"
            else:
                status = 'healthy'
                message = "Environment configuration valid"

            self.results.append(HealthCheckResult(
                component='environment_config',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='environment_config',
                status='critical',
                message=f"Environment check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_dependencies(self):
        """Check critical dependencies and versions."""
        start_time = time.time()

        try:
            dependencies = {}
            missing = []
            version_issues = []

            # Check critical imports
            critical_modules = {
                'numpy': '1.26.0',
                'gradio': '4.15.0',
                'torch': '2.0.0',
                'plotly': '5.17.0',
                'aiohttp': '3.9.0'
            }

            for module_name, min_version in critical_modules.items():
                try:
                    module = __import__(module_name)
                    version = getattr(module, '__version__', 'unknown')
                    dependencies[module_name] = version

                    # Simple version comparison
                    if version != 'unknown' and version < min_version:
                        version_issues.append(f"{module_name} {version} < {min_version}")

                except ImportError:
                    missing.append(module_name)

            # Check optional dependencies
            optional_modules = ['spaces', 'transformers', 'accelerate']
            for module_name in optional_modules:
                try:
                    module = __import__(module_name)
                    version = getattr(module, '__version__', 'unknown')
                    dependencies[module_name] = version
                except ImportError:
                    dependencies[module_name] = 'not_installed'

            details = {
                'dependencies': dependencies,
                'missing_critical': missing,
                'version_issues': version_issues,
                'optional_available': [m for m in optional_modules if dependencies.get(m) != 'not_installed']
            }

            if missing:
                status = 'critical'
                message = f"Missing critical dependencies: {', '.join(missing)}"
            elif version_issues:
                status = 'warning'
                message = f"Version issues: {', '.join(version_issues)}"
            else:
                status = 'healthy'
                message = f"All dependencies available: {len(dependencies)} modules"

            self.results.append(HealthCheckResult(
                component='dependencies',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='dependencies',
                status='critical',
                message=f"Dependency check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_web_service(self):
        """Check web service availability and responsiveness."""
        start_time = time.time()

        try:
            # Test basic connectivity
            response = requests.get(
                self.deployment_url,
                timeout=self.timeout,
                headers={'User-Agent': 'Felix-HealthChecker/1.0'}
            )

            response_time = time.time() - start_time

            details = {
                'url': self.deployment_url,
                'status_code': response.status_code,
                'response_time_seconds': response_time,
                'headers': dict(response.headers),
                'content_length': len(response.content)
            }

            if response.status_code == 200:
                if response_time < 5.0:
                    status = 'healthy'
                    message = f"Web service responsive: {response.status_code} in {response_time:.2f}s"
                else:
                    status = 'warning'
                    message = f"Web service slow: {response.status_code} in {response_time:.2f}s"
            else:
                status = 'critical'
                message = f"Web service error: HTTP {response.status_code}"

            self.results.append(HealthCheckResult(
                component='web_service',
                status=status,
                message=message,
                details=details,
                duration=response_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='web_service',
                status='critical',
                message=f"Web service check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_api_endpoints(self):
        """Check specific API endpoints."""
        start_time = time.time()

        endpoints = {
            '/health': 'GET',
            '/': 'GET'
        }

        endpoint_results = {}

        for endpoint, method in endpoints.items():
            try:
                url = f"{self.deployment_url.rstrip('/')}{endpoint}"
                response = requests.request(
                    method, url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'Felix-HealthChecker/1.0'}
                )

                endpoint_results[endpoint] = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'available': response.status_code < 400
                }

            except Exception as e:
                endpoint_results[endpoint] = {
                    'status_code': None,
                    'response_time': None,
                    'available': False,
                    'error': str(e)
                }

        # Determine overall status
        available_endpoints = sum(1 for r in endpoint_results.values() if r['available'])
        total_endpoints = len(endpoint_results)

        details = {
            'endpoints_tested': endpoint_results,
            'available_count': available_endpoints,
            'total_count': total_endpoints,
            'availability_percentage': (available_endpoints / total_endpoints) * 100
        }

        if available_endpoints == total_endpoints:
            status = 'healthy'
            message = f"All {total_endpoints} API endpoints available"
        elif available_endpoints > 0:
            status = 'warning'
            message = f"{available_endpoints}/{total_endpoints} API endpoints available"
        else:
            status = 'critical'
            message = "No API endpoints available"

        self.results.append(HealthCheckResult(
            component='api_endpoints',
            status=status,
            message=message,
            details=details,
            duration=time.time() - start_time
        ))

    async def _check_gradio_interface(self):
        """Check Gradio interface functionality."""
        start_time = time.time()

        try:
            # Test Gradio interface
            response = requests.get(
                self.deployment_url,
                timeout=self.timeout
            )

            content = response.text

            # Check for Gradio-specific elements
            gradio_indicators = [
                'gradio',
                'Felix Framework',
                'helix',
                'agent'
            ]

            found_indicators = [indicator for indicator in gradio_indicators if indicator.lower() in content.lower()]

            details = {
                'content_length': len(content),
                'gradio_indicators_found': found_indicators,
                'gradio_indicators_total': len(gradio_indicators),
                'likely_gradio_interface': len(found_indicators) >= 2
            }

            if len(found_indicators) >= 3:
                status = 'healthy'
                message = f"Gradio interface detected: {len(found_indicators)}/{len(gradio_indicators)} indicators"
            elif len(found_indicators) >= 1:
                status = 'warning'
                message = f"Partial Gradio interface: {len(found_indicators)}/{len(gradio_indicators)} indicators"
            else:
                status = 'critical'
                message = "Gradio interface not detected"

            self.results.append(HealthCheckResult(
                component='gradio_interface',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='gradio_interface',
                status='critical',
                message=f"Gradio interface check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_performance_baselines(self):
        """Check performance against baselines."""
        start_time = time.time()

        try:
            # Test helix computation performance
            from core.helix_geometry import HelixGeometry

            helix = HelixGeometry(33.0, 0.001, 100.0, 33)

            # Benchmark helix position calculation
            computation_start = time.time()
            positions = []
            for i in range(10000):
                t = i / 9999.0
                pos = helix.get_position_at_t(t)
                positions.append(pos)

            computation_time = time.time() - computation_start
            computation_rate = len(positions) / computation_time

            # Performance thresholds
            baseline_rate = 50000  # positions per second

            details = {
                'positions_computed': len(positions),
                'computation_time_seconds': computation_time,
                'computation_rate_per_second': computation_rate,
                'baseline_rate_per_second': baseline_rate,
                'performance_ratio': computation_rate / baseline_rate,
                'meets_baseline': computation_rate >= baseline_rate
            }

            if computation_rate >= baseline_rate:
                status = 'healthy'
                message = f"Performance above baseline: {computation_rate:.0f} pos/s (>{baseline_rate} pos/s)"
            elif computation_rate >= baseline_rate * 0.8:
                status = 'warning'
                message = f"Performance below baseline: {computation_rate:.0f} pos/s (<{baseline_rate} pos/s)"
            else:
                status = 'critical'
                message = f"Performance critically low: {computation_rate:.0f} pos/s (<<{baseline_rate} pos/s)"

            self.results.append(HealthCheckResult(
                component='performance_baselines',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='performance_baselines',
                status='critical',
                message=f"Performance baseline check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_gpu_availability(self):
        """Check GPU availability and ZeroGPU configuration."""
        start_time = time.time()

        try:
            gpu_info = {
                'torch_available': False,
                'cuda_available': False,
                'spaces_zerogpu': os.getenv('SPACES_ZERO_GPU') == 'true',
                'gpu_count': 0,
                'gpu_devices': []
            }

            # Check PyTorch and CUDA
            try:
                import torch
                gpu_info['torch_available'] = True
                gpu_info['cuda_available'] = torch.cuda.is_available()

                if gpu_info['cuda_available']:
                    gpu_info['gpu_count'] = torch.cuda.device_count()
                    for i in range(gpu_info['gpu_count']):
                        gpu_device = {
                            'id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_total': torch.cuda.get_device_properties(i).total_memory,
                            'memory_allocated': torch.cuda.memory_allocated(i) if torch.cuda.is_initialized() else 0
                        }
                        gpu_info['gpu_devices'].append(gpu_device)

            except ImportError:
                pass

            # Check Spaces GPU decorator
            try:
                import spaces
                gpu_info['spaces_module_available'] = True
            except ImportError:
                gpu_info['spaces_module_available'] = False

            details = gpu_info

            # Determine status
            if gpu_info['spaces_zerogpu'] and gpu_info['spaces_module_available']:
                status = 'healthy'
                message = "ZeroGPU configuration optimal"
            elif gpu_info['cuda_available']:
                status = 'healthy'
                message = f"CUDA available: {gpu_info['gpu_count']} GPU(s)"
            elif gpu_info['spaces_zerogpu']:
                status = 'warning'
                message = "ZeroGPU enabled but spaces module not available"
            else:
                status = 'warning'
                message = "No GPU acceleration available (CPU mode)"

            self.results.append(HealthCheckResult(
                component='gpu_availability',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='gpu_availability',
                status='warning',
                message=f"GPU check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_memory_usage(self):
        """Check memory usage patterns and optimization."""
        start_time = time.time()

        try:
            import gc

            # Force garbage collection
            gc.collect()

            # Get memory information
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            details = {
                'system_memory_total_gb': memory.total / (1024**3),
                'system_memory_available_gb': memory.available / (1024**3),
                'system_memory_percent': memory.percent,
                'process_memory_rss_gb': process_memory.rss / (1024**3),
                'process_memory_vms_gb': process_memory.vms / (1024**3),
                'gc_counts': gc.get_count()
            }

            # Memory thresholds
            if memory.percent < 70:
                status = 'healthy'
                message = f"Memory usage normal: {memory.percent:.1f}%"
            elif memory.percent < 85:
                status = 'warning'
                message = f"Memory usage elevated: {memory.percent:.1f}%"
            else:
                status = 'critical'
                message = f"Memory usage critical: {memory.percent:.1f}%"

            self.results.append(HealthCheckResult(
                component='memory_usage',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='memory_usage',
                status='warning',
                message=f"Memory check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_security_configuration(self):
        """Check security configuration and best practices."""
        start_time = time.time()

        try:
            security_issues = []
            security_warnings = []

            # Check environment variables
            sensitive_vars = ['HF_TOKEN', 'SECRET_KEY', 'API_KEY']
            for var in sensitive_vars:
                value = os.getenv(var)
                if value and len(value) < 20:
                    security_warnings.append(f"{var} appears to be a test/demo token")

            # Check file permissions (Unix-like systems)
            if os.name != 'nt':
                sensitive_files = ['.env', '.env.local', 'secrets.json']
                for filename in sensitive_files:
                    filepath = Path(filename)
                    if filepath.exists():
                        stat = filepath.stat()
                        mode = oct(stat.st_mode)[-3:]
                        if mode[-1] != '0':  # World readable
                            security_issues.append(f"{filename} is world-readable")

            # Check debug mode in production
            if os.getenv('FELIX_DEBUG') == 'true' and os.getenv('ENVIRONMENT') == 'production':
                security_warnings.append("Debug mode enabled in production")

            details = {
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
                'debug_enabled': os.getenv('FELIX_DEBUG') == 'true',
                'security_issues': security_issues,
                'security_warnings': security_warnings
            }

            if security_issues:
                status = 'critical'
                message = f"Security issues found: {', '.join(security_issues)}"
            elif security_warnings:
                status = 'warning'
                message = f"Security warnings: {', '.join(security_warnings)}"
            else:
                status = 'healthy'
                message = "Security configuration appears secure"

            self.results.append(HealthCheckResult(
                component='security_configuration',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='security_configuration',
                status='warning',
                message=f"Security check failed: {e}",
                duration=time.time() - start_time
            ))

    async def _check_secrets_management(self):
        """Check secrets management configuration."""
        start_time = time.time()

        try:
            secrets_status = {
                'hf_token_available': bool(os.getenv('HF_TOKEN')),
                'environment_set': bool(os.getenv('ENVIRONMENT')),
                'debug_configured': os.getenv('FELIX_DEBUG') is not None,
                'secrets_in_env': True,  # Assume true if running
                'secrets_in_files': False
            }

            # Check for secrets in files (security risk)
            secret_files = ['.env', '.env.local', 'config.json', 'secrets.json']
            for filename in secret_files:
                if Path(filename).exists():
                    secrets_status['secrets_in_files'] = True
                    break

            details = secrets_status

            if secrets_status['hf_token_available'] and not secrets_status['secrets_in_files']:
                status = 'healthy'
                message = "Secrets properly managed via environment variables"
            elif secrets_status['secrets_in_files']:
                status = 'warning'
                message = "Secrets found in files - should use environment variables"
            elif not secrets_status['hf_token_available']:
                status = 'warning'
                message = "HF_TOKEN not available - LLM features disabled"
            else:
                status = 'healthy'
                message = "Basic secrets management in place"

            self.results.append(HealthCheckResult(
                component='secrets_management',
                status=status,
                message=message,
                details=details,
                duration=time.time() - start_time
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component='secrets_management',
                status='warning',
                message=f"Secrets management check failed: {e}",
                duration=time.time() - start_time
            ))

    def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Calculate overall status
        statuses = [result.status for result in self.results]

        if 'critical' in statuses:
            overall_status = 'critical'
        elif 'warning' in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'

        # Component summary
        component_summary = {}
        for result in self.results:
            component_summary[result.component] = {
                'status': result.status,
                'message': result.message,
                'duration': result.duration
            }

        # Statistics
        total_checks = len(self.results)
        healthy_count = sum(1 for r in self.results if r.status == 'healthy')
        warning_count = sum(1 for r in self.results if r.status == 'warning')
        critical_count = sum(1 for r in self.results if r.status == 'critical')

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'deployment_url': self.deployment_url,
                'total_checks': total_checks,
                'check_duration_seconds': sum(r.duration or 0 for r in self.results)
            },
            'overall_status': overall_status,
            'summary': {
                'healthy': healthy_count,
                'warning': warning_count,
                'critical': critical_count,
                'health_percentage': (healthy_count / total_checks) * 100 if total_checks > 0 else 0
            },
            'components': component_summary,
            'detailed_results': [result.to_dict() for result in self.results],
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []

        # Analyze results for specific recommendations
        for result in self.results:
            if result.status == 'critical':
                if result.component == 'web_service':
                    recommendations.append("Check deployment and restart web service")
                elif result.component == 'felix_core':
                    recommendations.append("Verify Felix Framework installation and dependencies")
                elif result.component == 'dependencies':
                    recommendations.append("Install missing dependencies with pip install -r requirements.txt")
                elif result.component == 'system_health':
                    recommendations.append("Check system resources and consider scaling")

            elif result.status == 'warning':
                if result.component == 'performance_baselines':
                    recommendations.append("Consider optimizing performance or scaling resources")
                elif result.component == 'gpu_availability':
                    recommendations.append("Enable ZeroGPU for optimal performance")
                elif result.component == 'security_configuration':
                    recommendations.append("Review and address security warnings")
                elif result.component == 'memory_usage':
                    recommendations.append("Monitor memory usage and consider optimization")

        # General recommendations
        if not recommendations:
            recommendations.append("All systems operational - continue monitoring")

        return recommendations

async def main():
    """Main health check execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Felix Framework Deployment Health Check')
    parser.add_argument('--url', default='http://localhost:7860',
                       help='Deployment URL to check')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Request timeout in seconds')
    parser.add_argument('--output', default='health-check-report.json',
                       help='Output file for health report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create health checker
    checker = DeploymentHealthChecker(
        deployment_url=args.url,
        timeout=args.timeout
    )

    # Run health checks
    report = await checker.run_all_checks()

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"🏥 Felix Framework Health Check Report")
    print(f"📊 Overall Status: {report['overall_status'].upper()}")
    print(f"✅ Healthy: {report['summary']['healthy']}")
    print(f"⚠️  Warning: {report['summary']['warning']}")
    print(f"❌ Critical: {report['summary']['critical']}")
    print(f"📈 Health Score: {report['summary']['health_percentage']:.1f}%")
    print(f"📄 Full report saved to: {args.output}")

    if args.verbose:
        print(f"\n📋 Component Details:")
        for component, details in report['components'].items():
            print(f"  {component}: {details['status']} - {details['message']}")

        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    # Exit with appropriate code
    if report['overall_status'] == 'critical':
        sys.exit(1)
    elif report['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())