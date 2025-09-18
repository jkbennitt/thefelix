# Felix Framework - HuggingFace Spaces Deployment 🌪️

This document provides comprehensive instructions for deploying Felix Framework on HuggingFace Spaces with ZeroGPU optimization.

## Quick Deploy to HuggingFace Spaces

[![Deploy to HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-lg.svg)](https://huggingface.co/spaces?sdk=gradio&template=git&owner=YOUR_USERNAME&repo=https://github.com/CalebisGross/thefelix)

## 📋 Prerequisites

### 1. HuggingFace Account Setup
- Create account at [huggingface.co](https://huggingface.co)
- Generate API token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
- **Recommended**: Upgrade to HF Pro for ZeroGPU access and higher rate limits

### 2. Repository Preparation
```bash
git clone https://github.com/CalebisGross/thefelix.git
cd thefelix
```

### 3. Environment Variables
- `HF_TOKEN`: Your HuggingFace API token (required for full functionality)
- `FELIX_TOKEN_BUDGET`: Token budget for LLM usage (default: 50,000)
- `FELIX_DEBUG`: Enable debug logging (optional)

## 🚀 Deployment Methods

### Method 1: Direct Spaces Creation (Recommended)

1. **Create New Space**
   - Go to [HuggingFace Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose:
     - **SDK**: Gradio
     - **Hardware**: ZeroGPU (requires HF Pro)
     - **Visibility**: Public or Private

2. **Configure Space**
   ```yaml
   # Add to your Space's metadata (README.md header)
   title: Felix Framework
   emoji: 🌪️
   colorFrom: blue
   colorTo: purple
   sdk: gradio
   sdk_version: 4.15.0
   app_file: app.py
   pinned: false
   license: mit
   hardware: zero-gpu-medium
   ```

3. **Upload Files**
   - Upload all project files to your Space
   - Ensure `app.py`, `requirements-hf.txt`, and `src/` directory are included
   - Add `HF_TOKEN` to Space secrets (Settings > Repository Secrets)

### Method 2: Git Repository Integration

1. **Fork Repository**
   ```bash
   gh repo fork CalebisGross/thefelix --clone
   cd thefelix
   ```

2. **Create Space from Git**
   - Use "Create Space" with Git template
   - Point to your forked repository
   - Configure hardware as ZeroGPU

## ⚙️ Configuration Files

### requirements-hf.txt
```txt
# Core dependencies optimized for HF Spaces
spaces>=0.19.0
gradio>=4.15.0
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
plotly>=5.17.0
numpy>=1.26.0
scipy>=1.11.0
# ... (see complete file)
```

### app.py Features
- **ZeroGPU Integration**: `@spaces.GPU` decorators for compute-intensive operations
- **Real-time Progress**: `gr.Progress` for task processing updates
- **Mobile Responsive**: Modern CSS with mobile-first design
- **Interactive 3D**: Plotly-based helix visualization
- **Error Handling**: Comprehensive GPU resource management
- **Performance Monitoring**: Real-time system metrics

## 🎯 ZeroGPU Optimizations

### GPU Memory Management
```python
@spaces.GPU(duration=120)  # 2-minute GPU allocation
def process_with_gpu(self, task_description, agent_types, progress):
    # Automatic GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Process agents with GPU acceleration
    # ... processing logic

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
```

### Batch Processing
- Multiple agents processed simultaneously on GPU
- Intelligent batching based on memory availability
- Automatic fallback to CPU if GPU unavailable

### Memory Optimization
- Aggressive memory cleanup between operations
- Model caching with memory threshold monitoring
- Automatic GPU memory management

## 🎨 Interface Features

### 1. Interactive Demo Tab
- Task input with intelligent suggestions
- Agent type selection (research, analysis, synthesis, critic)
- Real-time 3D helix visualization
- GPU/CPU processing options
- Advanced configuration controls

### 2. 3D Helix Explorer
- Interactive visualization controls
- Camera preset options
- Agent position filtering
- Mathematical model information

### 3. Performance Dashboard
- Real-time system metrics
- GPU utilization monitoring
- Task history and success rates
- Resource usage tracking

### 4. Educational Content
- Framework comparison with LangGraph and mesh systems
- Research validation results
- Mathematical foundation explanations
- Interactive guided tours

### 5. Export & Sharing
- Multiple export formats (JSON, CSV, PNG)
- Session sharing capabilities
- Configurable privacy settings

## 📱 Mobile Responsiveness

### CSS Features
```css
/* Responsive design for all screen sizes */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    .main-header h1 {
        font-size: 2.2em;
    }
}

/* Dark mode support */
.dark .stats-card {
    background: #1e293b;
    border-color: #334155;
}
```

### Mobile Optimizations
- Touch-friendly interface elements
- Responsive grid layouts
- Optimized button sizes
- Swipeable tabs and controls

## 🔧 Troubleshooting

### Common Issues

#### 1. ZeroGPU Not Available
```
Error: ZeroGPU not available
```
**Solution**:
- Upgrade to HuggingFace Pro account
- Select ZeroGPU hardware in Space settings
- Check `SPACES_ZERO_GPU` environment variable

#### 2. GPU Memory Issues
```
Error: CUDA out of memory
```
**Solution**:
- Reduce `max_agents` parameter
- Enable "Aggressive Memory Optimization"
- Use smaller model variants

#### 3. HuggingFace Token Issues
```
Error: Invalid or missing HF_TOKEN
```
**Solution**:
- Add `HF_TOKEN` to Space secrets
- Verify token has proper permissions
- Check token hasn't expired

#### 4. Model Loading Failures
```
Error: Failed to load model to GPU
```
**Solution**:
- Check model availability on HuggingFace Hub
- Verify model supports the selected hardware
- Try fallback to Inference API mode

### Performance Optimization Tips

1. **Hardware Selection**
   - Use `zero-gpu-medium` for most use cases
   - Upgrade to `zero-gpu-large` for complex tasks
   - Consider `cpu-upgrade` for CPU-only deployment

2. **Memory Management**
   - Enable batch processing for efficiency
   - Use aggressive memory optimization
   - Limit concurrent agent operations

3. **Model Selection**
   - Prefer quantized models for GPU deployment
   - Use smaller models for faster inference
   - Balance quality vs. speed requirements

## 📊 Monitoring and Analytics

### Built-in Metrics
- Task completion rates
- Response times by agent type
- GPU utilization and memory usage
- Error rates and types
- User interaction patterns

### Health Checks
```python
# Automatic health monitoring
def health_check():
    return {
        "status": "healthy",
        "zerogpu_status": "available",
        "components": {
            "helix_geometry": "operational",
            "agents": "operational",
            "llm_integration": "operational"
        }
    }
```

## 🛠️ Development Workflow

### Local Development
```bash
# Setup local environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-hf.txt

# Run locally
python app.py
```

### Testing Before Deployment
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Validate Felix Framework
python validate_felix_framework.py

# Test GPU functionality (if available)
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### Deployment Checklist
- [ ] All dependencies in `requirements-hf.txt`
- [ ] `HF_TOKEN` configured in Space secrets
- [ ] ZeroGPU hardware selected
- [ ] Mobile responsiveness tested
- [ ] Error handling verified
- [ ] Performance benchmarks completed

## 🌟 Advanced Features

### Custom Model Integration
```python
# Add custom models to model_configs
CUSTOM_MODELS = {
    ModelType.RESEARCH: HFModelConfig(
        model_id="microsoft/DialoGPT-large",
        use_zerogpu=True,
        batch_size=2
    )
}
```

### API Endpoints
```python
# Health check endpoint
GET /health -> {"status": "healthy", ...}

# System information
GET /system-info -> {"platform": "...", "gpu_info": {...}}
```

### Event Tracking
- User interactions
- Task completion analytics
- Performance metrics
- Error logging

## 🔗 Additional Resources

- **GitHub Repository**: https://github.com/CalebisGross/thefelix
- **Research Documentation**: [RESEARCH_LOG.md](RESEARCH_LOG.md)
- **Mathematical Model**: [Mathematical Model Documentation](docs/architecture/core/mathematical_model.md)
- **HuggingFace Spaces Documentation**: https://huggingface.co/docs/hub/spaces
- **ZeroGPU Guide**: https://huggingface.co/docs/hub/spaces-gpus

## 📞 Support

For deployment issues or questions:
1. Check the [GitHub Issues](https://github.com/CalebisGross/thefelix/issues)
2. Review [Troubleshooting](#troubleshooting) section
3. Contact the development team

---

**Ready to deploy Felix Framework and explore the future of multi-agent AI coordination!** 🚀