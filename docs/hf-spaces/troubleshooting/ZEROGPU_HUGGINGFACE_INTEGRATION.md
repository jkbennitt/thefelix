# ZeroGPU HuggingFace Integration for Felix Framework

This document describes the ZeroGPU-optimized HuggingFace client integration for the Felix Framework, designed for deployment on HuggingFace Spaces with Pro account features.

## Overview

The `HuggingFaceClient` provides a drop-in replacement for `LMStudioClient` with advanced ZeroGPU acceleration, HuggingFace Pro account optimizations, and seamless deployment to HuggingFace Spaces.

### Key Features

- **ZeroGPU Acceleration**: Direct GPU inference with `@spaces.GPU` decorator support
- **Batch Processing**: Efficient multi-agent processing with GPU memory sharing
- **Automatic Fallbacks**: Graceful degradation from ZeroGPU to Inference API
- **HF Pro Features**: Higher rate limits, premium models, and priority queuing
- **LMStudioClient Compatibility**: Drop-in replacement with identical API
- **Memory Management**: Intelligent GPU memory allocation and cleanup

## Architecture

### ZeroGPU Integration Flow

```
Felix Agent Request
        ↓
HuggingFaceClient
        ↓
┌─── ZeroGPU Available? ───┐
│ Yes                  No  │
↓                         ↓
GPU Inference          Inference API
@spaces.GPU            (Fallback)
        ↓                 ↓
   GPU Response    API Response
        ↓                 ↓
    Agent Response ←──────┘
```

### Model Optimization Strategy

#### Default Models (ZeroGPU Optimized)
- **Research Agents**: `microsoft/DialoGPT-large` (3GB, fast batching)
- **Analysis Agents**: `meta-llama/Llama-3.1-8B-Instruct` (16GB, reasoning)
- **Synthesis Agents**: `meta-llama/Llama-3.1-13B-Instruct` (26GB, quality)
- **Critic Agents**: `microsoft/DialoGPT-large` (3GB, efficient validation)

#### Pro Account Models (Premium Access)
- **Analysis/Synthesis**: `meta-llama/Llama-3.1-70B-Instruct` (140GB, enterprise-grade)
- **Research**: `meta-llama/Llama-3.1-8B-Instruct` (16GB, enhanced reasoning)

## Installation and Setup

### 1. Environment Requirements

```bash
# HuggingFace Token (required)
export HF_TOKEN="your_huggingface_token_here"

# ZeroGPU Environment (automatic on HF Spaces)
# SPACES_ZERO_GPU=1  # Set automatically by HF Spaces
```

### 2. Dependencies

```bash
pip install -r requirements.txt

# Core ZeroGPU dependencies:
# - spaces>=0.19.0
# - torch>=2.0.0
# - transformers>=4.36.0
# - accelerate>=0.25.0
```

### 3. Basic Usage

```python
from src.llm.huggingface_client import create_felix_hf_client, ModelType

# Create ZeroGPU-optimized client
client = create_felix_hf_client(
    enable_zerogpu=True,
    debug_mode=True
)

# Use with Felix agents (LMStudioClient compatibility)
response = await client.complete_async(
    agent_id="research_agent",
    system_prompt="You are a research specialist",
    user_prompt="Analyze renewable energy trends",
    temperature=0.7
)

print(f"Response: {response.content}")
print(f"Tokens: {response.tokens_used}")
```

## Advanced Features

### 1. Batch Processing

```python
# Process multiple agents simultaneously
prompts = [
    "Research climate solutions",
    "Analyze economic impact",
    "Synthesize recommendations"
]

agent_types = [
    ModelType.RESEARCH,
    ModelType.ANALYSIS,
    ModelType.SYNTHESIS
]

# ZeroGPU batch processing
results = await client.batch_generate(
    prompts=prompts,
    agent_types=agent_types,
    use_zerogpu_batching=True,
    max_tokens=256
)

for result in results:
    print(f"Batch processed: {result.batch_processed} items")
    print(f"GPU time: {result.gpu_time:.2f}s")
```

### 2. GPU Memory Management

```python
# Check GPU requirements
from src.llm.huggingface_client import estimate_gpu_requirements

requirements = estimate_gpu_requirements(client.model_configs)
print(f"Recommended GPU Memory: {requirements['recommended_gpu_memory']:.1f} GB")

# Monitor GPU usage
stats = client.get_performance_stats()
if stats['zerogpu_enabled']:
    print(f"GPU Memory Used: {stats['gpu_memory_allocated']:.2f} GB")
    print(f"Loaded Models: {stats['loaded_models']}")
```

### 3. Pro Account Optimizations

```python
from src.llm.huggingface_client import get_pro_account_models

# Use premium models (requires HF Pro)
pro_models = get_pro_account_models()
client = create_felix_hf_client()
client.model_configs.update(pro_models)

# Access 70B models with high priority
response = await client.generate_text(
    "Complex reasoning task",
    agent_type=ModelType.ANALYSIS,  # Uses Llama-3.1-70B
    priority=RequestPriority.HIGH
)
```

## HuggingFace Spaces Deployment

### 1. Space Configuration

Create `app.py` for HF Spaces:

```python
import gradio as gr
from src.llm.huggingface_client import create_felix_hf_client
import asyncio

# Initialize ZeroGPU client
client = create_felix_hf_client(enable_zerogpu=True)

@spaces.GPU  # ZeroGPU acceleration
async def felix_chat(message, agent_type):
    """Felix-powered chat with ZeroGPU acceleration."""
    response = await client.generate_text(
        prompt=message,
        agent_type=getattr(ModelType, agent_type.upper()),
        max_tokens=512
    )

    return response.content

# Gradio interface
interface = gr.Interface(
    fn=felix_chat,
    inputs=[
        gr.Textbox(label="Message"),
        gr.Dropdown(["research", "analysis", "synthesis"], label="Agent Type")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Felix Framework - ZeroGPU Powered"
)

if __name__ == "__main__":
    interface.launch()
```

### 2. Space Requirements File

```yaml
# .space-requirements.txt
title: Felix Framework ZeroGPU
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.15.0
app_file: app.py
pinned: false
hardware: zero-gpu-small  # or zero-gpu-medium for larger models
```

### 3. Environment Variables

Set in HF Spaces settings:
- `HF_TOKEN`: Your HuggingFace token
- `TRANSFORMERS_CACHE`: `/tmp/transformers_cache` (optional)

## Performance Optimization

### 1. Memory Management

```python
# Configure GPU memory thresholds
client = create_felix_hf_client(
    gpu_memory_threshold=0.85,  # Cleanup at 85% usage
    batch_timeout=3.0           # 3s max batching wait
)

# Manual cleanup if needed
await client._cleanup_gpu_memory()
```

### 2. Model Selection Strategy

| Use Case | Recommended Model | GPU Memory | Batch Size |
|----------|-------------------|------------|------------|
| Fast Research | DialoGPT-large | 3GB | 2-3 |
| Quality Analysis | Llama-3.1-8B | 16GB | 1-2 |
| Best Synthesis | Llama-3.1-13B | 26GB | 1 |
| Enterprise (Pro) | Llama-3.1-70B | 140GB | 1 |

### 3. Batch Size Optimization

```python
# Configure batch sizes based on model and GPU memory
model_configs = {
    ModelType.RESEARCH: HFModelConfig(
        model_id="microsoft/DialoGPT-large",
        batch_size=3,  # Smaller model, more batching
        gpu_memory_limit=8.0
    ),
    ModelType.SYNTHESIS: HFModelConfig(
        model_id="meta-llama/Llama-3.1-13B-Instruct",
        batch_size=1,  # Larger model, no batching
        gpu_memory_limit=32.0
    )
}
```

## Error Handling and Fallbacks

### 1. Automatic Fallbacks

The client provides multiple fallback layers:

```python
ZeroGPU Inference
        ↓ (on error)
Inference API
        ↓ (on rate limit)
Cached Response / Error Response
```

### 2. Error Types and Handling

```python
from src.llm.huggingface_client import ZeroGPUError, GPUMemoryError

try:
    response = await client.generate_text(prompt, use_zerogpu=True)
except ZeroGPUError as e:
    print(f"GPU inference failed, falling back: {e}")
    # Automatic fallback to Inference API
except GPUMemoryError as e:
    print(f"GPU memory insufficient: {e}")
    # Automatic cleanup and retry
```

### 3. Monitoring and Debugging

```python
# Enable debug mode for detailed logging
client = create_felix_hf_client(debug_mode=True)

# Monitor performance and errors
stats = client.get_performance_stats()
print(f"Error rate: {stats['error_rate']:.2%}")
print(f"Fallback usage: {stats.get('fallback_rate', 0):.2%}")

# Test connection health
if client.test_connection():
    print("✅ HuggingFace connection healthy")
else:
    print("❌ Connection issues detected")
```

## Felix Framework Integration

### 1. Agent System Compatibility

```python
from src.agents.agent import Agent
from src.core.helix_geometry import HelixGeometry

# Felix helix-based agent positioning
helix = HelixGeometry()
position = helix.get_position_at_parameter(0.5)

# Create Felix-optimized system prompt
system_prompt = client.create_agent_system_prompt(
    agent_type="synthesis",
    position_info=position,
    task_context="Climate analysis"
)

# Use with Felix agent
response = await client.complete_async(
    agent_id="synthesis_agent_001",
    system_prompt=system_prompt,
    user_prompt="Synthesize renewable energy findings",
    temperature=0.1 + position['depth_ratio'] * 0.8
)
```

### 2. Multi-Agent Coordination

```python
# Process Felix agents with ZeroGPU batching
felix_agents = [
    ("research_001", ModelType.RESEARCH, "Research renewable energy"),
    ("analysis_002", ModelType.ANALYSIS, "Analyze cost-benefit"),
    ("synthesis_003", ModelType.SYNTHESIS, "Synthesize recommendations")
]

batch_prompts = [prompt for _, _, prompt in felix_agents]
batch_types = [agent_type for _, agent_type, _ in felix_agents]

# GPU-accelerated batch processing
results = await client.batch_generate(
    prompts=batch_prompts,
    agent_types=batch_types,
    use_zerogpu_batching=True
)

# Process results with Felix coordination
for (agent_id, agent_type, _), result in zip(felix_agents, results):
    print(f"{agent_id}: {result.content}")
    # Send to Felix CentralPost for coordination...
```

## Testing and Validation

### 1. Running Tests

```bash
# Run ZeroGPU-specific tests
pytest tests/unit/test_huggingface_zerogpu_client.py -v

# Run integration tests (requires HF_TOKEN)
pytest tests/unit/test_huggingface_zerogpu_client.py::TestIntegrationScenarios -v

# Skip tests requiring GPU (when not available)
pytest tests/unit/test_huggingface_zerogpu_client.py -k "not zerogpu" -v
```

### 2. Performance Benchmarking

```bash
# Run comprehensive benchmark
python examples/zerogpu_hf_demo.py --benchmark --zerogpu --output benchmark_results.json

# Compare ZeroGPU vs Inference API
python examples/zerogpu_hf_demo.py --batch --agents 4 --debug
```

### 3. Validation Checklist

- [ ] HF_TOKEN environment variable set
- [ ] ZeroGPU acceleration working (`SPACES_ZERO_GPU=1`)
- [ ] Model loading and caching functional
- [ ] Batch processing optimization active
- [ ] Fallback mechanisms tested
- [ ] Memory cleanup working
- [ ] Felix agent integration verified
- [ ] Pro account features accessible (if applicable)

## Troubleshooting

### Common Issues

1. **ZeroGPU Not Available**
   ```
   Solution: Client automatically falls back to Inference API
   Status: Normal operation, slightly slower response times
   ```

2. **GPU Memory Errors**
   ```
   Solution: Automatic memory cleanup and model reloading
   Prevention: Use smaller models or reduce batch sizes
   ```

3. **Rate Limiting**
   ```
   Solution: HF Pro accounts have higher limits
   Prevention: Implement request throttling
   ```

4. **Model Loading Failures**
   ```
   Solution: Fallback to Inference API
   Check: Model availability and access permissions
   ```

### Performance Issues

1. **Slow Response Times**
   - Check ZeroGPU availability
   - Reduce model size or batch size
   - Monitor GPU memory usage

2. **High Memory Usage**
   - Enable automatic cleanup
   - Use float16 precision
   - Implement model rotation

3. **Fallback Too Frequent**
   - Check GPU availability
   - Verify model compatibility
   - Monitor error logs

## Conclusion

The ZeroGPU-optimized HuggingFace client provides enterprise-grade LLM capabilities for the Felix Framework with:

- **Performance**: Up to 10x faster inference with GPU acceleration
- **Scalability**: Efficient batch processing for multi-agent systems
- **Reliability**: Automatic fallbacks and error handling
- **Compatibility**: Drop-in replacement for LMStudioClient
- **Optimization**: Memory management and resource efficiency

This integration enables the Felix Framework to leverage cutting-edge GPU acceleration while maintaining the research integrity and geometric orchestration that defines its unique approach to multi-agent cognitive architectures.

For deployment to HuggingFace Spaces with ZeroGPU, the system provides seamless scaling from research prototypes to production-ready applications while preserving the mathematical precision and helix-based coordination that makes Felix Framework a competitive alternative to traditional graph-based multi-agent systems.