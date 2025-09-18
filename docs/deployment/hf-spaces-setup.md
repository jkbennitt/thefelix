# Felix Framework Deployment Guide: Hugging Face Spaces with ZeroGPU

> **Complete guide for deploying Felix Framework on Hugging Face Spaces with ZeroGPU acceleration**

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Deployment](#quick-deployment)
- [Step-by-Step Setup](#step-by-step-setup)
- [ZeroGPU Configuration](#zerogpu-configuration)
- [Environment Variables](#environment-variables)
- [Model Selection](#model-selection)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

### Hugging Face Account Requirements
- **Hugging Face Pro Account** (required for ZeroGPU access)
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

## Quick Deployment

### 1. One-Click Deployment
**[🚀 Deploy Felix Framework to Your Space](https://huggingface.co/spaces/CalebisGross/felix-framework?duplicate=true)**

This will create a copy of the Felix Framework in your Hugging Face account with all dependencies configured.

### 2. Repository Clone Method
```bash
# Clone the repository
git clone https://github.com/CalebisGross/thefelix.git
cd thefelix

# Verify HF Spaces configuration
ls app.py gradio_interface.py requirements.txt
```

## Step-by-Step Setup

### Step 1: Create New Hugging Face Space

1. **Go to Hugging Face Spaces**: https://huggingface.co/spaces
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
├── app.py                    # Main Gradio application
├── gradio_interface.py       # UI components and demos
├── requirements.txt          # Dependencies optimized for ZeroGPU
├── README.md                 # Space description and usage
├── src/                      # Core Felix Framework code
│   ├── core/
│   │   └── helix_geometry.py
│   ├── agents/
│   │   └── specialized_agents.py
│   ├── communication/
│   │   └── central_post.py
│   └── llm/
│       └── hf_transformers_client.py
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

# Copy Felix Framework files
cp -r ../thefelix/src .
cp -r ../thefelix/config .
cp ../thefelix/app.py .
cp ../thefelix/gradio_interface.py .
cp ../thefelix/requirements.txt .

# Commit and push
git add .
git commit -m "Deploy Felix Framework with ZeroGPU support"
git push
```

**Option B: Web Interface Upload**
1. Use HF Spaces file upload interface
2. Upload files individually or as zip
3. Ensure proper directory structure is maintained

### Step 3: Configure ZeroGPU Settings

#### In your Space settings:
```yaml
# Space configuration
title: "Felix Framework - Helix-Based Multi-Agent System"
emoji: "🌪️"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.15.0"
python_version: "3.10"
hardware: zero-gpu
suggested_hardware: zero-gpu
```

#### Add ZeroGPU decorators to compute-intensive functions:
```python
# In your app.py
import spaces

@spaces.GPU
def generate_helix_points(num_turns=33, nodes=133):
    """GPU-accelerated helix geometry computation"""
    # Your helix generation code here
    pass

@spaces.GPU
def run_multi_agent_blog_writer(topic, complexity="medium"):
    """GPU-accelerated multi-agent processing"""
    # Your blog writing logic here
    pass
```

## ZeroGPU Configuration

### GPU Memory Management
```python
# config/hf_spaces_config.json
{
  "zerogpu": {
    "memory_limit": "24GB",
    "timeout": 120,
    "auto_cleanup": true,
    "model_cache_size": "8GB"
  },
  "models": {
    "research_agent": "microsoft/DialoGPT-medium",
    "analysis_agent": "google/flan-t5-large",
    "synthesis_agent": "microsoft/DialoGPT-large"
  },
  "helix_params": {
    "num_turns": 33,
    "nodes": 133,
    "radius_start": 33.0,
    "radius_end": 0.001
  }
}
```

### Performance Optimization Settings
```python
# GPU optimization in app.py
import torch
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    device = "cpu"

# Memory-efficient model loading
from transformers import AutoTokenizer, AutoModelForCausalLM

@spaces.GPU
def load_model_optimized(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto",          # Automatic device placement
        low_cpu_mem_usage=True      # Reduce CPU memory usage
    )
    return model
```

## Environment Variables

### Required Environment Variables

Set these in your Hugging Face Space settings:

```bash
# Space Settings > Variables and secrets

# Model Configuration
HF_TOKEN=hf_your_token_here                    # Your Hugging Face token
MODEL_CACHE_DIR=/tmp/model_cache               # Model caching directory
TRANSFORMERS_CACHE=/tmp/transformers_cache    # Transformers cache

# Felix Framework Configuration
FELIX_DEBUG=false                             # Debug mode (use true for development)
FELIX_LOG_LEVEL=INFO                          # Logging level
HELIX_PRECISION=1e-12                         # Mathematical precision
MAX_AGENTS=20                                 # Maximum concurrent agents

# Performance Settings
BATCH_SIZE=4                                  # Processing batch size
MAX_LENGTH=512                                # Maximum token length
TEMPERATURE_RESEARCH=0.9                      # Research agent creativity
TEMPERATURE_SYNTHESIS=0.1                     # Synthesis agent precision

# ZeroGPU Settings
GPU_MEMORY_FRACTION=0.8                       # GPU memory allocation
ENABLE_MIXED_PRECISION=true                   # Mixed precision training
TORCH_COMPILE=false                           # PyTorch compilation (experimental)
```

### Optional Environment Variables
```bash
# Advanced Configuration
ENABLE_GRADIO_ANALYTICS=false                 # Gradio usage analytics
CUSTOM_CSS_PATH=static/custom.css             # Custom CSS styling
HELIX_VISUALIZATION_FPS=30                    # Visualization frame rate
AGENT_SPAWN_DELAY=0.1                         # Delay between agent spawns (seconds)
```

## Model Selection

### Recommended Models for ZeroGPU

#### Small Models (Fast Response)
- **microsoft/DialoGPT-medium** - 345M parameters, good for research agents
- **google/flan-t5-base** - 250M parameters, excellent for analysis tasks
- **distilbert-base-uncased** - 66M parameters, fast text processing

#### Medium Models (Balanced)
- **microsoft/DialoGPT-large** - 774M parameters, high-quality synthesis
- **google/flan-t5-large** - 780M parameters, strong reasoning capabilities
- **facebook/bart-large** - 406M parameters, excellent summarization

#### Large Models (High Quality)
- **microsoft/DialoGPT-large** - For synthesis agents requiring high quality
- **google/flan-t5-xl** - 3B parameters, advanced reasoning (requires careful memory management)

### Model Loading Strategy
```python
# Efficient model management for ZeroGPU
class FelixModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    @spaces.GPU
    def load_agent_model(self, agent_type):
        """Load model based on agent specialization"""
        model_mapping = {
            "research": "microsoft/DialoGPT-medium",   # Fast exploration
            "analysis": "google/flan-t5-base",         # Balanced reasoning
            "synthesis": "microsoft/DialoGPT-large",   # High quality output
            "critic": "google/flan-t5-base"            # Validation tasks
        }

        model_name = model_mapping.get(agent_type, "microsoft/DialoGPT-medium")

        if model_name not in self.models:
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

        return self.models[model_name], self.tokenizers[model_name]
```

## Performance Optimization

### GPU Memory Optimization
```python
# Memory-efficient processing
@spaces.GPU
def process_with_memory_management(text, model, tokenizer):
    """Process text with automatic memory cleanup"""
    try:
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Use attention slicing for large sequences
        if hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing()

        # Process in batches if needed
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model.generate(
                inputs.input_ids,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    finally:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Helix Computation Optimization
```python
# GPU-accelerated helix geometry
@spaces.GPU
def generate_helix_points_gpu(num_turns=33, nodes=133):
    """Generate helix points using GPU acceleration"""
    import torch

    # Use GPU tensors for computation
    t = torch.linspace(0, 2*math.pi*num_turns, nodes, device='cuda')

    # Radius tapering calculation on GPU
    radius = 33.0 * torch.exp(-t / (2*math.pi*num_turns) * math.log(33.0/0.001))

    # Helix coordinates
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = t / (2*math.pi) * 10  # Height scaling

    # Return as numpy for compatibility
    return torch.stack([x, y, z], dim=1).cpu().numpy()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ZeroGPU Memory Errors
**Error**: "CUDA out of memory"
```python
# Solution: Implement proper memory management
@spaces.GPU
def memory_safe_processing(data):
    try:
        # Your processing code
        result = process_data(data)
        return result
    finally:
        # Always clear cache
        torch.cuda.empty_cache()

# Use smaller batch sizes
BATCH_SIZE = 2  # Reduce from default 4

# Enable memory mapping for large models
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={0: "20GB"}  # Limit GPU memory usage
)
```

#### 2. Model Loading Failures
**Error**: "Model not found" or "Connection timeout"
```python
# Solution: Robust model loading with fallbacks
def load_model_with_fallback(model_name, fallback_model="microsoft/DialoGPT-medium"):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            resume_download=True  # Resume interrupted downloads
        )
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print(f"Loading fallback model: {fallback_model}")
        return AutoModelForCausalLM.from_pretrained(fallback_model)
```

#### 3. Gradio Interface Issues
**Error**: "Interface not loading" or "Component errors"
```python
# Solution: Progressive interface loading
def create_gradio_interface():
    """Create interface with error handling"""
    try:
        # Main interface
        interface = gr.Interface(
            fn=felix_blog_writer,
            inputs=[
                gr.Textbox(label="Topic", placeholder="Enter blog topic..."),
                gr.Slider(1, 10, value=5, label="Complexity Level")
            ],
            outputs=[
                gr.Textbox(label="Generated Blog"),
                gr.Plot(label="Helix Visualization")
            ],
            title="🌪️ Felix Framework - Helix-Based Multi-Agent System",
            description="Experience geometric multi-agent coordination"
        )
        return interface
    except Exception as e:
        # Fallback minimal interface
        return gr.Interface(
            fn=lambda x: f"Error: {e}",
            inputs=gr.Textbox(),
            outputs=gr.Textbox()
        )
```

#### 4. Performance Issues
**Problem**: Slow response times or timeouts
```python
# Solution: Optimize for ZeroGPU constraints
@spaces.GPU(duration=120)  # Explicit timeout
def optimized_blog_writer(topic):
    """Optimized version for ZeroGPU"""

    # Use smaller models for speed
    model_name = "microsoft/DialoGPT-medium"  # Instead of large

    # Reduce computation complexity
    helix_points = generate_helix_points_gpu(
        num_turns=20,  # Reduced from 33
        nodes=50       # Reduced from 133
    )

    # Parallel agent processing with memory limits
    agents = create_agents(max_agents=3)  # Reduced from 5+

    return process_topic_optimized(topic, agents, helix_points)
```

### Debug Mode Configuration
```python
# Enable debug mode for troubleshooting
if os.getenv("FELIX_DEBUG", "false").lower() == "true":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Enable detailed GPU monitoring
    if torch.cuda.is_available():
        torch.cuda.memory._record_memory_history()

    # Enable Gradio debug mode
    gr.Interface.launch(debug=True, show_error=True)
```

## Advanced Configuration

### Custom ZeroGPU Decorators
```python
# Custom GPU decorator with monitoring
def felix_gpu(duration=60, memory_limit="20GB"):
    """Custom GPU decorator for Felix Framework"""
    def decorator(func):
        @spaces.GPU(duration=duration)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Performance logging
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                print(f"Function {func.__name__}:")
                print(f"  Duration: {end_time - start_time:.2f}s")
                print(f"  Memory usage: {(end_memory - start_memory) / 1024**2:.1f}MB")

                torch.cuda.empty_cache()
        return wrapper
    return decorator

# Usage
@felix_gpu(duration=90, memory_limit="24GB")
def complex_helix_computation():
    # Your GPU-intensive code here
    pass
```

### Multi-User Load Balancing
```python
# Handle concurrent users efficiently
from threading import Lock
import queue

class FelixLoadBalancer:
    def __init__(self, max_concurrent=5):
        self.max_concurrent = max_concurrent
        self.request_queue = queue.Queue()
        self.active_requests = 0
        self.lock = Lock()

    @spaces.GPU
    def process_request(self, user_input):
        """Process user request with load balancing"""
        with self.lock:
            if self.active_requests >= self.max_concurrent:
                return "Server busy. Please try again in a moment."
            self.active_requests += 1

        try:
            result = self.felix_process(user_input)
            return result
        finally:
            with self.lock:
                self.active_requests -= 1
```

### Production Monitoring
```python
# Add monitoring and analytics
import time
from collections import defaultdict

class FelixMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def log_request(self, endpoint, duration, success=True):
        """Log request metrics"""
        self.metrics['requests'].append({
            'endpoint': endpoint,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })

    def get_stats(self):
        """Return performance statistics"""
        total_requests = len(self.metrics['requests'])
        if total_requests == 0:
            return "No requests yet"

        avg_duration = sum(r['duration'] for r in self.metrics['requests']) / total_requests
        success_rate = sum(1 for r in self.metrics['requests'] if r['success']) / total_requests
        uptime = time.time() - self.start_time

        return f"""
        📊 Felix Framework Stats:
        • Total Requests: {total_requests}
        • Average Response: {avg_duration:.2f}s
        • Success Rate: {success_rate:.1%}
        • Uptime: {uptime/3600:.1f} hours
        """
```

## Deployment Checklist

### Pre-Deployment
- [ ] **Hugging Face Pro account** activated
- [ ] **ZeroGPU access** confirmed
- [ ] **Repository structure** verified
- [ ] **Dependencies** tested locally
- [ ] **Model compatibility** checked

### Deployment
- [ ] **Space created** with correct settings
- [ ] **Files uploaded** maintaining directory structure
- [ ] **Environment variables** configured
- [ ] **ZeroGPU decorators** added to compute functions
- [ ] **Requirements.txt** optimized for HF Spaces

### Post-Deployment
- [ ] **Interface loads** without errors
- [ ] **Demo functions** working correctly
- [ ] **GPU allocation** functioning properly
- [ ] **Performance metrics** within acceptable range
- [ ] **Error handling** implemented
- [ ] **Monitoring** enabled for production use

## Support and Resources

### Felix Framework Documentation
- **[Core Architecture Guide](../architecture/core/mathematical_model.md)**
- **[LLM Integration Guide](../guides/llm-integration/LLM_INTEGRATION.md)**
- **[Development Rules](../guides/development/DEVELOPMENT_RULES.md)**

### Hugging Face Resources
- **[ZeroGPU Documentation](https://huggingface.co/docs/hub/spaces-gpus)**
- **[Gradio Documentation](https://gradio.app/docs/)**
- **[Spaces Examples](https://huggingface.co/spaces)**

### Community
- **[Felix Framework Discussions](https://github.com/CalebisGross/thefelix/discussions)**
- **[Hugging Face Discord](https://discord.com/invite/JfAtkvEtRb)**
- **[Issues and Bug Reports](https://github.com/CalebisGross/thefelix/issues)**

---

**Ready to deploy Felix Framework? Start with the [Quick Deployment](#quick-deployment) section above!**

*For questions or issues, please see our [troubleshooting section](#troubleshooting) or open an issue on GitHub.*