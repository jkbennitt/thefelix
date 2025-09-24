# Secrets Management for Felix Framework

This document outlines the secure management of secrets and environment variables for Felix Framework deployment on HuggingFace Spaces with ZeroGPU integration.

## Overview

Felix Framework requires several secrets and configuration variables for proper operation, particularly for HuggingFace API integration, deployment automation, and monitoring. This guide ensures secure handling of sensitive information.

## Required Secrets

### GitHub Repository Secrets

Configure the following secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

#### Core HuggingFace Integration

| Secret Name | Description | Required | Example Value |
|------------|-------------|----------|---------------|
| `HF_TOKEN` | HuggingFace Pro API token | ✅ **Required** | `hf_xxxxxxxxxxxxxxxxxxxxx` |
| `HF_SPACE_ID` | Production HF Spaces ID | ✅ **Required** | `jkbennitt/felix-framework` |
| `HF_SPACE_ID_STAGING` | Staging HF Spaces ID | 🔶 **Optional** | `jkbennitt/felix-framework-staging` |

#### Extended Configuration (Optional)

| Secret Name | Description | Required | Example Value |
|------------|-------------|----------|---------------|
| `HF_ORG_TOKEN` | Organization-level token | 🔶 **Optional** | `hf_xxxxxxxxxxxxxxxxxxxxx` |
| `DISCORD_WEBHOOK_URL` | Deployment notifications | 🔶 **Optional** | `https://discord.com/api/webhooks/...` |
| `SLACK_WEBHOOK_URL` | Slack notifications | 🔶 **Optional** | `https://hooks.slack.com/services/...` |

## HuggingFace Token Setup

### Obtaining HF Pro Token

1. **Visit HuggingFace Settings**
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Sign in to your HuggingFace Pro account

2. **Create New Token**
   - Click "New token"
   - Name: `felix-framework-deployment`
   - Type: **Write** (required for Spaces deployment)
   - Scopes needed:
     - ✅ `repo` - Repository access
     - ✅ `inference` - Inference API access
     - ✅ `spaces` - Spaces deployment

3. **Token Permissions**
   ```json
   {
     "role": "write",
     "scopes": ["repo", "inference", "spaces"],
     "expires": null,
     "description": "Felix Framework deployment token"
   }
   ```

### HF Pro Account Benefits

Felix Framework leverages HF Pro account features:

- **Higher Rate Limits**: 10,000 requests/hour vs 1,000 for free accounts
- **Priority Queue**: Faster inference processing
- **ZeroGPU Access**: Required for GPU acceleration features
- **Private Spaces**: Option for private deployments
- **Advanced Models**: Access to larger models (13B+ parameters)
- **Extended Quotas**: Higher token budgets for LLM usage

## Space Configuration

### Creating HuggingFace Spaces

#### Production Space Setup

```bash
# Using HuggingFace CLI
huggingface-cli repo create jkbennitt/felix-framework --type space --space_sdk gradio

# Space metadata (automatically set by deployment)
# Title: Felix Framework - ZeroGPU
# SDK: gradio
# Python Version: 3.12
# App File: app.py
```

#### Staging Space Setup (Optional)

```bash
huggingface-cli repo create jkbennitt/felix-framework-staging --type space --space_sdk gradio
```

### Space Visibility Options

| Visibility | Usage | Access |
|------------|-------|--------|
| **Public** | Production deployment | Anyone can access |
| **Unlisted** | Staging/testing | Only with direct link |
| **Private** | Development | Organization members only |

## Environment Variables

### HuggingFace Spaces Environment

These are automatically set by HF Spaces infrastructure:

```bash
# Automatically provided by HF Spaces
SPACE_ID="jkbennitt/felix-framework"
SPACE_AUTHOR_NAME="jkbennitt"
SPACES_ZERO_GPU="true"  # Indicates ZeroGPU availability
HF_HOME="/tmp/.cache/huggingface"  # HF cache directory
```

### Felix Framework Configuration

Set these in your Space settings or deployment configuration:

```bash
# Core configuration
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"  # From secrets
FELIX_DEBUG="false"  # Enable debug logging
FELIX_TOKEN_BUDGET="50000"  # Token budget for LLM usage

# Performance tuning
FELIX_MAX_CONCURRENT_AGENTS="10"  # Limit concurrent agents
FELIX_GPU_MEMORY_THRESHOLD="0.9"  # GPU memory cleanup threshold
FELIX_BATCH_TIMEOUT="5.0"  # Max wait time for batching

# Monitoring and analytics
FELIX_ENABLE_METRICS="true"  # Enable performance metrics
FELIX_ANALYTICS_ENDPOINT=""  # Optional analytics endpoint
```

## Security Best Practices

### Token Security

1. **Never Commit Tokens**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo ".env.local" >> .gitignore
   echo "*.token" >> .gitignore
   echo "hf_token.txt" >> .gitignore
   ```

2. **Use Environment Variables**
   ```python
   import os

   # ✅ Correct
   hf_token = os.getenv('HF_TOKEN')

   # ❌ Never do this
   hf_token = "hf_xxxxxxxxxxxxxxxxxxxxx"
   ```

3. **Token Rotation**
   - Rotate tokens every 90 days
   - Immediately rotate if compromise suspected
   - Use different tokens for different environments

### Access Control

1. **Repository Settings**
   - Limit who can modify secrets
   - Require admin approval for secret changes
   - Enable audit logging

2. **Organization Policies**
   ```yaml
   # .github/settings.yml
   repository:
     allow_secrets: true
     secrets_policy: "admin_only"
     require_two_factor: true
   ```

3. **Secret Scanning**
   - Enable GitHub secret scanning
   - Use pre-commit hooks to prevent token commits
   - Regular audit of committed history

### Deployment Security

1. **Secure Deployment Pipeline**
   ```yaml
   # GitHub Actions security
   permissions:
     contents: read
     id-token: write  # For OIDC
     actions: read
   ```

2. **Environment Isolation**
   - Separate tokens for staging/production
   - Different Space IDs for different environments
   - Isolated testing environments

3. **Monitoring and Alerting**
   - Monitor token usage
   - Alert on unusual API activity
   - Track deployment failures

## Local Development

### Development Environment Setup

1. **Create .env.local file** (never commit):
   ```bash
   # .env.local (DO NOT COMMIT)
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   FELIX_DEBUG=true
   FELIX_TOKEN_BUDGET=10000
   SPACES_ZERO_GPU=false  # For local development
   ```

2. **Load Environment Variables**
   ```python
   from dotenv import load_dotenv

   # Load development environment
   load_dotenv('.env.local')
   ```

3. **Testing Without Tokens**
   ```python
   # Felix runs in demo mode without HF_TOKEN
   # All functionality available except LLM integration
   if not os.getenv('HF_TOKEN'):
       print("Running in demo mode - LLM features disabled")
   ```

## Troubleshooting

### Common Issues

#### Invalid Token Error
```bash
# Check token validity
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami
```

#### Insufficient Permissions
```json
{
  "error": "Insufficient permissions",
  "details": "Token needs 'spaces' scope for deployment"
}
```

**Solution**: Regenerate token with correct scopes.

#### Rate Limiting
```json
{
  "error": "Rate limit exceeded",
  "reset_time": "2024-12-19T15:30:00Z"
}
```

**Solution**: Upgrade to HF Pro or wait for reset.

#### ZeroGPU Access Denied
```bash
# Error: ZeroGPU not available for this account
```

**Solution**: Ensure HF Pro subscription is active.

### Debugging Commands

```bash
# Test HF CLI authentication
huggingface-cli whoami

# Check Space status
huggingface-cli repo info spaces/jkbennitt/felix-framework

# Test deployment pipeline
github-actions-cli workflow run "HF Spaces ZeroGPU Deployment"

# Validate secrets configuration
python scripts/validate_hf_config.py
```

## Monitoring and Maintenance

### Token Usage Monitoring

```python
# scripts/monitor_token_usage.py
import requests

def check_token_usage(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://huggingface.co/api/usage", headers=headers)

    if response.status_code == 200:
        usage = response.json()
        print(f"API calls: {usage['requests_count']}")
        print(f"Rate limit: {usage['rate_limit']}")
        print(f"Reset time: {usage['reset_time']}")
    else:
        print(f"Error checking usage: {response.status_code}")
```

### Automated Secret Rotation

```yaml
# .github/workflows/rotate-secrets.yml
name: Rotate Secrets
on:
  schedule:
    - cron: '0 0 1 * *'  # First day of each month
  workflow_dispatch:

jobs:
  rotate-hf-token:
    runs-on: ubuntu-latest
    steps:
      - name: Notify of upcoming rotation
        run: |
          echo "HF Token rotation scheduled for next maintenance window"
          # Send notification to team
```

### Health Monitoring

```python
# Continuous monitoring script
async def monitor_deployment_health():
    space_url = "https://jkbennitt-felix-framework.hf.space"

    # Check space accessibility
    response = await aiohttp.get(f"{space_url}/health")
    if response.status != 200:
        await send_alert("Felix Framework space is down")

    # Check ZeroGPU status
    gpu_status = await check_zerogpu_availability()
    if not gpu_status:
        await send_alert("ZeroGPU not available")
```

## Compliance and Audit

### Audit Trail

1. **GitHub Actions Logs**
   - All deployment activities logged
   - Secret access tracked
   - Failure reasons recorded

2. **HuggingFace Activity**
   - API usage tracking
   - Space deployment history
   - Model access logs

3. **Security Events**
   - Token creation/rotation
   - Permission changes
   - Unusual access patterns

### Compliance Checklist

- [ ] All secrets stored in GitHub Secrets (not in code)
- [ ] Tokens have minimum required permissions
- [ ] Regular token rotation schedule established
- [ ] Environment isolation implemented
- [ ] Audit logging enabled
- [ ] Emergency procedures documented
- [ ] Team trained on security practices
- [ ] Backup access methods established

## Emergency Procedures

### Token Compromise

1. **Immediate Actions**
   ```bash
   # Revoke compromised token
   huggingface-cli token revoke $OLD_TOKEN

   # Generate new token
   huggingface-cli token create --name "felix-emergency-$(date +%s)"
   ```

2. **Update Secrets**
   - Update GitHub repository secrets
   - Redeploy affected services
   - Audit access logs

3. **Communication**
   - Notify team of incident
   - Document timeline and actions
   - Review security procedures

### Service Outage

1. **Diagnosis**
   ```bash
   # Check Space status
   curl -I https://jkbennitt-felix-framework.hf.space

   # Check HF API status
   curl -I https://api-inference.huggingface.co/status
   ```

2. **Escalation**
   - Contact HuggingFace support
   - Activate backup deployment
   - Notify users of service status

This secrets management system ensures secure, automated deployment of Felix Framework while maintaining the highest security standards for API tokens and sensitive configuration data.