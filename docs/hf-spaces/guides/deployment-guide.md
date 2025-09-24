# Felix Framework HuggingFace Spaces Deployment Guide

> **Complete guide for deploying Felix Framework on HuggingFace Spaces with ZeroGPU acceleration**

## 🚀 Quick Deploy Options

### Option 1: One-Click Duplicate (Recommended)
**[🚀 Deploy Felix Framework to Your Space](https://huggingface.co/spaces/jkbennitt/felix-framework?duplicate=true)**

This creates a copy of Felix Framework in your HF account with all dependencies configured.

### Option 2: Manual Setup
Follow the detailed setup instructions below for custom configuration.

---

## 📋 Prerequisites

### HuggingFace Account Requirements
- **HuggingFace Pro Account** (required for ZeroGPU access)
- **Verified email address**
- **Access to Spaces feature**

### Technical Requirements
- **Python 3.9+** (tested with 3.10-3.12)
- **ZeroGPU compatibility** (automatic GPU allocation)
- **Git LFS support** for large model files

### Felix Framework Features
- **Helix-based multi-agent coordination**
- **Real-time agent visualization**
- **Interactive blog writing demos**
- **Statistical validation tools**

---

## 🔧 Step-by-Step Manual Setup

### Step 1: Create New HuggingFace Space

1. **Go to HuggingFace Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Configure Space settings**:
   ```
   Space name: felix-framework-demo
   License: MIT
   SDK: Gradio
   Hardware: ZeroGPU (requires Pro account)
   Python version: 3.10
   ```
4. **Set to Public** (recommended for demo purposes)

### Step 2: Upload Felix Framework Files

#### Essential Files for HF Spaces:
```
felix-framework-space/
├── README.md                 # Space description with YAML frontmatter
├── app.py                    # Main Gradio application
├── requirements.txt          # Dependencies optimized for ZeroGPU
├── src/                      # Core Felix Framework code
│   ├── core/
│   │   └── helix_geometry.py
│   ├── agents/
│   │   └── specialized_agents.py
│   ├── communication/
│   │   └── central_post.py
│   └── llm/
│       └── huggingface_client.py
├── config/
│   └── hf_spaces_config.json
└── examples/
    └── blog_writer_hf.py
```

#### Upload Methods:

**Option A: Git Upload**
```bash
# Clone your empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/felix-framework-demo
cd felix-framework-demo

# Copy Felix Framework files from thefelix repository
cp -r ../thefelix/src .
cp ../thefelix/app.py .
cp ../thefelix/requirements.txt .
cp ../thefelix/README.md .

# Add and commit
git add .
git commit -m "feat: add Felix Framework with ZeroGPU support"
git push
```

**Option B: Web Interface Upload**
1. Go to your Space's "Files" tab
2. Upload files individually or drag-and-drop folders
3. Ensure all required files are present

### Step 3: Configure Environment Variables

In your Space settings (`Settings > Repository secrets`), add:

```bash
# Required
HF_TOKEN=your_huggingface_token_here

# Optional
FELIX_TOKEN_BUDGET=50000
FELIX_DEBUG=false
```

### Step 4: Verify Deployment

1. **Check build logs** in your Space's "Logs" tab
2. **Wait for successful startup** (typically 2-3 minutes)
3. **Test the interface** by running a demo task
4. **Verify ZeroGPU allocation** in the performance dashboard

---

## ⚡ ZeroGPU Configuration

### Hardware Selection
- **zero-gpu-medium** (recommended): 16GB VRAM, good for most tasks
- **zero-gpu-large** (advanced): 24GB VRAM, for complex multi-agent tasks

### GPU Optimization Features
- **@spaces.GPU decorators** for compute-intensive operations
- **Automatic memory management** with intelligent cleanup
- **Batch processing** for multiple agents on single GPU allocation
- **Fallback mechanisms** to CPU when GPU unavailable

### Memory Management
```python
# Example ZeroGPU optimization in app.py
@spaces.GPU(duration=120)  # 2-minute GPU allocation
def process_with_gpu(task_description, agent_types, progress):
    # Automatic GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Process agents with GPU acceleration
    # ... processing logic

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
```

---

## 🛠️ Environment Variables

### Required Variables
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_TOKEN` | HuggingFace API token with write permissions | None | ✅ Yes |

### Optional Variables
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FELIX_TOKEN_BUDGET` | Token budget for LLM usage | 50000 | ❌ No |
| `FELIX_DEBUG` | Enable debug logging | false | ❌ No |
| `FELIX_MAX_AGENTS` | Maximum concurrent agents | 20 | ❌ No |

---

## 📊 Model Selection

### Supported Models
- **microsoft/DialoGPT-large**: Research and analysis tasks
- **meta-llama/Llama-3.1-8B-Instruct**: General purpose reasoning
- **meta-llama/Llama-3.1-13B-Instruct**: High-quality synthesis
- **Qwen/Qwen2.5-7B-Instruct**: Fast response generation

### Model Configuration
Models are automatically selected based on:
- **Agent type** (Research, Analysis, Synthesis, Critic)
- **Available GPU memory**
- **Task complexity level**
- **Token budget constraints**

---

## 🎯 Performance Optimization

### ZeroGPU Best Practices
1. **Use batch processing** for multiple agents
2. **Enable aggressive memory optimization**
3. **Choose appropriate complexity levels**
4. **Monitor GPU utilization** in dashboard

### Memory Optimization Tips
- **Reduce max_agents** if experiencing memory issues
- **Use smaller model variants** for faster inference
- **Enable model caching** for repeated operations
- **Clear GPU memory** between complex tasks

---

## 🔍 Troubleshooting

### Common Issues

#### ZeroGPU Not Available
```
Error: ZeroGPU not available
```
**Solution**:
- Upgrade to HuggingFace Pro account
- Select ZeroGPU hardware in Space settings
- Check `SPACES_ZERO_GPU` environment variable

#### GPU Memory Issues
```
Error: CUDA out of memory
```
**Solution**:
- Reduce `max_agents` parameter
- Enable "Aggressive Memory Optimization"
- Use smaller model variants

#### HuggingFace Token Issues
```
Error: Invalid or missing HF_TOKEN
```
**Solution**:
- Add `HF_TOKEN` to Space secrets
- Verify token has proper permissions
- Check token hasn't expired

#### Model Loading Failures
```
Error: Failed to load model to GPU
```
**Solution**:
- Check model availability on HuggingFace Hub
- Verify model supports the selected hardware
- Try fallback to Inference API mode

---

## 🔧 Advanced Configuration

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

---

## 📚 Additional Resources

- **[Felix Framework Repository](https://github.com/jkbennitt/thefelix)**
- **[Research Documentation](./RESEARCH_LOG.md)**
- **[Mathematical Model](./docs/architecture/core/mathematical_model.md)**
- **[HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)**
- **[ZeroGPU Guide](https://huggingface.co/docs/hub/spaces-gpus)**

---

**Ready to deploy Felix Framework and explore the future of multi-agent AI coordination!** 🚀