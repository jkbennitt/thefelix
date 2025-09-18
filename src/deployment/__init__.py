"""
Deployment utilities for Felix Framework.

This module provides specialized tools and utilities for deploying Felix Framework
in various environments, with particular focus on cloud platforms like HuggingFace
Spaces with ZeroGPU infrastructure.

Key Components:
- ZeroGPU monitoring and resource management
- Error handling and fallback strategies
- Batch processing optimization
- Deployment validation and configuration
- Performance optimization utilities
"""

from .zerogpu_monitor import (
    ZeroGPUMonitor,
    ResourceAlert,
    GPUMemorySnapshot,
    PerformanceMetrics,
    ResourceType,
    AlertSeverity,
    create_zerogpu_monitor,
    optimize_for_zerogpu
)

from .zerogpu_error_handler import (
    ZeroGPUErrorHandler,
    ErrorType,
    FallbackStrategy,
    ErrorContext,
    RecoveryAction,
    create_zerogpu_error_handler,
    setup_global_error_handling
)

from .batch_optimizer import (
    ZeroGPUBatchOptimizer,
    BatchTask,
    BatchRequest,
    BatchResult,
    BatchStrategy,
    AgentPriority,
    GPUResourceState,
    create_zerogpu_batch_optimizer
)

from .zerogpu_deployment_guide import (
    ZeroGPUDeploymentValidator,
    DeploymentReport,
    DeploymentCheck,
    DeploymentStatus,
    validate_zerogpu_deployment,
    create_deployment_package
)

__all__ = [
    # Monitoring
    'ZeroGPUMonitor',
    'ResourceAlert',
    'GPUMemorySnapshot',
    'PerformanceMetrics',
    'ResourceType',
    'AlertSeverity',
    'create_zerogpu_monitor',
    'optimize_for_zerogpu',

    # Error Handling
    'ZeroGPUErrorHandler',
    'ErrorType',
    'FallbackStrategy',
    'ErrorContext',
    'RecoveryAction',
    'create_zerogpu_error_handler',
    'setup_global_error_handling',

    # Batch Processing
    'ZeroGPUBatchOptimizer',
    'BatchTask',
    'BatchRequest',
    'BatchResult',
    'BatchStrategy',
    'AgentPriority',
    'GPUResourceState',
    'create_zerogpu_batch_optimizer',

    # Deployment Validation
    'ZeroGPUDeploymentValidator',
    'DeploymentReport',
    'DeploymentCheck',
    'DeploymentStatus',
    'validate_zerogpu_deployment',
    'create_deployment_package'
]