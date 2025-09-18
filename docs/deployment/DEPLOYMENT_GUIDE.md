# Felix Framework Cloud Deployment Guide

This guide provides comprehensive instructions for deploying the Felix Framework to production environments, with specific focus on Hugging Face Spaces.

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd thefelix
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Docker Compose (Recommended)**
   ```bash
   docker-compose up -d
   ```
   - Felix Framework: http://localhost:7860
   - Grafana: http://localhost:3000 (admin/felix-admin)
   - Prometheus: http://localhost:9090

3. **Manual Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt -r requirements-deployment.txt
   python -m deployment.web_service
   ```

## 🏭 Production Deployment

### Hugging Face Spaces

1. **Prepare Repository**
   ```bash
   # Ensure all files are ready
   cp .env.example .env
   # Configure production environment variables
   ```

2. **GitHub Secrets Configuration**
   Set these secrets in your GitHub repository:
   ```
   HF_TOKEN=your_huggingface_token
   HF_SPACE_ID=your-username/felix-framework
   ```

3. **Deploy via GitHub Actions**
   - Push to `main` branch triggers automatic deployment
   - Monitor deployment in GitHub Actions tab
   - Space will be available at `https://your-username-felix-framework.hf.space`

4. **Manual HF Spaces Deployment**
   ```bash
   # Install HF CLI
   pip install huggingface_hub[cli]

   # Login and create space
   huggingface-cli login
   huggingface-cli repo create your-username/felix-framework --type space --space_sdk docker

   # Push to space
   git remote add hf https://huggingface.co/spaces/your-username/felix-framework
   git push hf main
   ```

### Docker Production Deployment

1. **Build Production Image**
   ```bash
   docker build -t felix-framework:latest .
   ```

2. **Production Docker Compose**
   ```bash
   # Copy production compose file
   cp docker-compose.prod.yml docker-compose.yml

   # Start production stack
   docker-compose up -d
   ```

3. **Health Check**
   ```bash
   curl -f http://localhost:7860/health
   ```

## ⚙️ Configuration

### Environment Variables

#### Required Configuration
```bash
# Application
ENVIRONMENT=production
SECRET_KEY=your-super-secret-jwt-key-here
FELIX_API_KEY=your-felix-api-key

# LLM Integration (choose one)
LLM_ENDPOINT=http://localhost:1234  # LM Studio
OPENAI_API_KEY=your-openai-key      # OpenAI
HF_INFERENCE_API_KEY=your-hf-key    # Hugging Face
```

#### Optional Configuration
```bash
# Monitoring
SENTRY_DSN=your-sentry-dsn
ENABLE_METRICS=true

# Security
CORS_ORIGINS=["https://yourdomain.com"]
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Performance
MAX_MEMORY_USAGE_MB=1024
WORKERS=2
```

### Configuration Validation
```bash
python config/settings.py
```

## 🔒 Security Configuration

### API Keys and Secrets

1. **Generate Secure Keys**
   ```bash
   # JWT Secret Key
   openssl rand -hex 32

   # API Key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Environment-specific Secrets**
   - Development: Store in `.env` file
   - Production: Use environment variables or secrets management
   - Hugging Face Spaces: Set in Space settings

### Rate Limiting
- Anonymous users: 30 requests/minute
- Authenticated users: 60 requests/minute
- Premium users: 180 requests/minute

### CORS Configuration
```bash
# Development
CORS_ORIGINS=["http://localhost:3000", "http://localhost:7860"]

# Production
CORS_ORIGINS=["https://yourdomain.com"]
```

## 📊 Monitoring and Observability

### Health Checks
- `/health` - Basic health check
- `/health/detailed` - Comprehensive diagnostics
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Kubernetes readiness probe

### Metrics Collection
- Prometheus metrics: `/api/v1/metrics`
- Custom metrics for agents, LLM usage, and performance
- Grafana dashboards for visualization

### Logging
- Structured JSON logging in production
- Correlation IDs for request tracking
- Performance metrics collection
- Error tracking with Sentry (optional)

### Alerts Configuration
```bash
# Memory usage alert
felix_memory_usage_percent > 80

# High error rate alert
felix_error_rate_percent > 5

# Agent processing time alert
felix_agent_processing_seconds > 30
```

## 🚀 Performance Optimization

### Resource Limits
```bash
# Hugging Face Spaces constraints
MAX_MEMORY_GB=16
MAX_CPU_CORES=8
TIMEOUT_SECONDS=300

# Docker resource limits
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
```

### Caching Strategy
- Redis for session management and API caching
- In-memory caching for helix calculations
- Response caching for static endpoints

### Database Optimization
- SQLite for development
- PostgreSQL for production (if persistent storage needed)
- Connection pooling and query optimization

## 🔧 Troubleshooting

### Common Issues

1. **LLM Connection Failed**
   ```bash
   # Check endpoint
   curl http://localhost:1234/v1/models

   # Verify configuration
   python -c "from src.llm.lm_studio_client import LMStudioClient; print(LMStudioClient().test_connection())"
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   curl http://localhost:7860/health/detailed

   # Adjust limits
   export MAX_MEMORY_USAGE_MB=512
   ```

3. **Rate Limiting Issues**
   ```bash
   # Check rate limit status
   curl -H "X-API-Key: your-key" http://localhost:7860/api/v1/status
   ```

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug endpoints
export ENABLE_DEBUG_ENDPOINTS=true
```

### Log Analysis
```bash
# View logs
docker-compose logs -f felix-app

# Search for errors
grep ERROR logs/felix.log

# Performance analysis
grep "processing_time" logs/felix.log | jq .duration_seconds
```

## 📈 Scaling and High Availability

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  felix-app:
    deploy:
      replicas: 3

  nginx:
    deploy:
      replicas: 2
```

### Load Balancing
- Nginx upstream configuration
- Session affinity for WebSocket connections
- Health check integration

### Database Scaling
- Read replicas for analytics
- Connection pooling
- Query optimization

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
- Code quality checks (Black, flake8, mypy)
- Security scanning (Bandit, Safety)
- Comprehensive testing (unit, integration, performance)
- Docker image building and security scanning
- Automated deployment to Hugging Face Spaces

### Release Management
```bash
# Create release
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions will:
# 1. Run full test suite
# 2. Build and scan Docker image
# 3. Deploy to production
# 4. Create release artifacts
```

## 📚 API Documentation

### Core Endpoints
- `GET /` - Service information and status
- `POST /api/v1/process` - Process tasks with multi-agent coordination
- `GET /api/v1/agents` - List active agents
- `GET /api/v1/helix` - Helix visualization data
- `WebSocket /ws` - Real-time updates

### Authentication
```bash
# Login
curl -X POST http://localhost:7860/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "felix-admin-2024"}'

# Use token
curl -H "Authorization: Bearer your-jwt-token" \
  http://localhost:7860/api/v1/agents
```

### Example API Usage
```bash
# Process task
curl -X POST http://localhost:7860/api/v1/process \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "task": "Analyze the benefits of renewable energy",
    "use_multi_agent": true,
    "parameters": {"max_tokens": 1000}
  }'
```

## 🆘 Support and Maintenance

### Regular Maintenance
- Weekly security audits via GitHub Actions
- Monthly dependency updates
- Quarterly performance reviews
- Log rotation and cleanup

### Backup Strategy
- Configuration files in version control
- Database backups (if using persistent storage)
- Metrics and log archival

### Monitoring Checklist
- [ ] Health endpoints responding
- [ ] Error rates within acceptable limits
- [ ] Memory usage under threshold
- [ ] LLM integration functional
- [ ] WebSocket connections stable

---

## 📞 Getting Help

- **Documentation**: Check `/docs` endpoint (development only)
- **Health Status**: Monitor `/health/detailed` endpoint
- **Metrics**: Review Grafana dashboards
- **Logs**: Check structured logs for detailed information

For deployment-specific issues, refer to the comprehensive error handling and monitoring built into the Felix Framework.