# Felix Framework - ZeroGPU Optimized Dockerfile for HuggingFace Spaces
# Multi-stage build for optimized production deployment

# Build stage for dependency compilation
FROM python:3.12-slim as builder

# Set build-time variables
ARG TORCH_VERSION=2.0.1
ARG CUDA_VERSION=cu118

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch with CUDA support for ZeroGPU
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim as runtime

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    PORT=7860

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Essential system libraries
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    # CUDA runtime libraries (for ZeroGPU)
    libcudnn8 \
    # Network utilities
    curl \
    wget \
    # Process monitoring
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r felix && useradd -r -g felix -m -s /bin/bash felix

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application files with proper ownership
COPY --chown=felix:felix . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/tmp && \
    chown -R felix:felix /app/logs /app/cache /app/tmp

# Create performance monitoring directories
RUN mkdir -p /app/metrics /app/benchmarks && \
    chown -R felix:felix /app/metrics /app/benchmarks

# Verify Felix Framework core components
RUN python -c "
import sys
sys.path.insert(0, '/app')

try:
    from src.core.helix_geometry import HelixGeometry
    helix = HelixGeometry(33.0, 0.001, 100.0, 33)
    pos = helix.get_position_at_t(0.5)
    print(f'✅ Felix core validation successful: position {pos}')
except Exception as e:
    print(f'❌ Felix core validation failed: {e}')
    sys.exit(1)
"

# Verify ZeroGPU compatibility
RUN python -c "
import torch
import sys

print(f'🔧 PyTorch version: {torch.__version__}')
print(f'🔧 CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'🎮 CUDA version: {torch.version.cuda}')
    print(f'🎮 GPU count: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA not available in container (normal for build stage)')

# Test spaces import
try:
    import spaces
    print('✅ Spaces module available for ZeroGPU')
except ImportError:
    print('⚠️  Spaces module not available (will use mock in development)')

print('🌪️ Felix Framework Docker build completed successfully')
"

# Create startup script for health monitoring
RUN cat > /app/startup.sh << 'EOF'
#!/bin/bash
set -e

echo "🌪️ Starting Felix Framework..."
echo "Environment: $ENVIRONMENT"
echo "Port: $PORT"
echo "ZeroGPU: ${SPACES_ZERO_GPU:-false}"

# Health check function
health_check() {
    python -c "
    import sys
    sys.path.insert(0, '/app')
    from app import health_check
    result = health_check()
    if result['status'] == 'healthy':
        print('✅ Health check passed')
        exit(0)
    else:
        print('❌ Health check failed')
        exit(1)
    "
}

# Start background health monitoring
(
    while true; do
        sleep 30
        health_check || echo "⚠️ Health check warning at $(date)"
    done
) &

# Log system information
python -c "
import sys
sys.path.insert(0, '/app')
from app import get_system_info
import json
info = get_system_info()
print('🔍 System Information:')
print(json.dumps(info, indent=2))
"

# Start the application
exec python app.py "$@"
EOF

RUN chmod +x /app/startup.sh

# Switch to non-root user
USER felix

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port for HuggingFace Spaces
EXPOSE ${PORT}

# Labels for container metadata
LABEL org.opencontainers.image.title="Felix Framework - ZeroGPU"
LABEL org.opencontainers.image.description="Helix-based multi-agent cognitive architecture with ZeroGPU acceleration"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Felix Framework Contributors"
LABEL org.opencontainers.image.url="https://github.com/CalebisGross/thefelix"
LABEL org.opencontainers.image.source="https://github.com/CalebisGross/thefelix"
LABEL org.opencontainers.image.vendor="Felix Framework"
LABEL org.opencontainers.image.licenses="MIT"

# Performance optimization labels
LABEL felix.framework.version="1.0.0"
LABEL felix.zerogpu.enabled="true"
LABEL felix.optimization.level="production"
LABEL felix.architecture="helix-based"

# Set default command
CMD ["/app/startup.sh"]