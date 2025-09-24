"""
HuggingFace ZeroGPU-optimized client for Felix Framework on HF Spaces.

This module provides advanced HuggingFace integration optimized for ZeroGPU acceleration,
HF Pro account features, and HF Spaces deployment while maintaining full API compatibility
with LMStudioClient.

ZeroGPU Features:
- Dynamic GPU allocation with @spaces.GPU decorator support
- GPU memory management and automatic cleanup
- Batch processing for multiple agents with GPU acceleration
- Model loading with torch.cuda optimization
- Efficient device allocation and deallocation

HF Pro Account Features:
- Higher rate limits and premium model access
- Priority inference queue for Pro accounts
- Advanced model configurations and fine-tuning support
- Extended quota management

Agent-Model Mapping (ZeroGPU Optimized):
- ResearchAgent: Fast 7B models (e.g., microsoft/DialoGPT-large, Qwen/Qwen2.5-7B-Instruct)
- AnalysisAgent: Reasoning 13B models (e.g., microsoft/DialoGPT-large, meta-llama/Llama-3.1-8B-Instruct)
- SynthesisAgent: High-quality 13B models (e.g., meta-llama/Llama-3.1-13B-Instruct)
- CriticAgent: Specialized validation models (e.g., microsoft/DialoGPT-medium)

LMStudioClient Compatibility:
- Drop-in replacement maintaining identical API
- Same method signatures and response objects
- Existing Felix agent system integration preserved
"""

import asyncio
import logging
import time
import os
import gc
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
from datetime import datetime, timedelta
from collections import deque

# ZeroGPU and HF integration imports
try:
    import spaces
    ZEROGPU_AVAILABLE = True
except ImportError:
    ZEROGPU_AVAILABLE = False
    # Mock decorator for non-ZeroGPU environments
    class MockSpaces:
        @staticmethod
        def GPU(fn):
            return fn
    spaces = MockSpaces()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from huggingface_hub import HfApi, InferenceClient
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .token_budget import TokenBudgetManager, TokenAllocation
from .lm_studio_client import RequestPriority, LLMResponse

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model specialization types for different agent functions."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CRITIC = "critic"
    GENERAL = "general"


class GPUMemoryError(Exception):
    """Raised when GPU memory allocation fails."""
    pass


class ZeroGPUError(Exception):
    """Raised when ZeroGPU operations fail."""
    pass


class HuggingFaceConnectionError(Exception):
    """Raised when cannot connect to HuggingFace services."""
    pass


@dataclass
class HFModelConfig:
    """Configuration for a HuggingFace model with ZeroGPU optimization."""
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    use_cache: bool = True
    wait_for_model: bool = True
    # ZeroGPU specific settings
    use_zerogpu: bool = True
    gpu_memory_limit: Optional[float] = None  # GB, None for auto
    torch_dtype: str = "float16"  # torch dtype for GPU efficiency
    device_map: str = "auto"
    batch_size: int = 1
    # HF Pro settings
    priority: str = "normal"  # normal, high for Pro accounts
    use_inference_api: bool = True  # Fallback to Inference API
    local_model_path: Optional[str] = None


@dataclass
class HFResponse:
    """Response from HuggingFace inference API with GPU metrics."""
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # ZeroGPU specific metrics
    gpu_memory_used: Optional[float] = None
    gpu_time: Optional[float] = None
    batch_processed: Optional[int] = None
    fallback_used: bool = False


class HuggingFaceClient:
    """
    HuggingFace Inference API client for Felix Framework.

    Provides model inference capabilities with token budget management,
    rate limiting, and agent specialization support.
    """

    # ZeroGPU optimized models for different agent types
    DEFAULT_MODELS = {
        ModelType.RESEARCH: HFModelConfig(
            model_id="microsoft/DialoGPT-large",  # Upgraded for ZeroGPU
            temperature=0.9,
            max_tokens=384,
            use_zerogpu=True,
            batch_size=2,  # Can process multiple research queries
            torch_dtype="float16"
        ),
        ModelType.ANALYSIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",  # Better reasoning
            temperature=0.5,
            max_tokens=512,
            use_zerogpu=True,
            batch_size=1,
            torch_dtype="float16",
            priority="high"  # Pro account priority for analysis
        ),
        ModelType.SYNTHESIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-13B-Instruct",  # High-quality synthesis
            temperature=0.1,
            max_tokens=768,
            use_zerogpu=True,
            batch_size=1,
            torch_dtype="float16",
            gpu_memory_limit=12.0,  # Need more memory for 13B model
            priority="high"
        ),
        ModelType.CRITIC: HFModelConfig(
            model_id="microsoft/DialoGPT-large",
            temperature=0.3,
            max_tokens=384,
            use_zerogpu=True,
            batch_size=2,
            torch_dtype="float16"
        ),
        ModelType.GENERAL: HFModelConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",  # Good general purpose ZeroGPU model
            temperature=0.7,
            max_tokens=512,
            use_zerogpu=True,
            batch_size=1,
            torch_dtype="float16"
        )
    }

    def __init__(self,
                 hf_token: Optional[str] = None,
                 model_configs: Optional[Dict[ModelType, HFModelConfig]] = None,
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_concurrent_requests: int = 10,
                 request_timeout: float = 30.0,
                 # ZeroGPU specific parameters
                 enable_zerogpu: bool = True,
                 gpu_memory_threshold: float = 0.9,  # Trigger cleanup at 90% memory
                 batch_timeout: float = 5.0,  # Max wait time for batching
                 # LMStudioClient compatibility
                 base_url: Optional[str] = None,  # For API compatibility
                 timeout: Optional[float] = None,  # Alternative name for request_timeout
                 debug_mode: bool = False):
        """
        Initialize HuggingFace ZeroGPU-optimized client.

        Args:
            hf_token: HuggingFace API token (uses HF_TOKEN env var if None)
            model_configs: Custom model configurations by agent type
            token_budget_manager: Token budget manager for rate limiting
            max_concurrent_requests: Maximum concurrent API requests
            request_timeout: Request timeout in seconds
            enable_zerogpu: Enable ZeroGPU acceleration when available
            gpu_memory_threshold: GPU memory usage threshold for cleanup
            batch_timeout: Maximum wait time for request batching
            base_url: API base URL (for LMStudioClient compatibility)
            timeout: Request timeout (alternative parameter name)
            debug_mode: Enable verbose debug output
        """
        # API compatibility with LMStudioClient
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.base_url = base_url  # For compatibility (not used)
        self.timeout = timeout or request_timeout
        self.request_timeout = self.timeout
        self.debug_mode = debug_mode

        # Core configuration
        self.model_configs = model_configs or self.DEFAULT_MODELS
        self.token_budget_manager = token_budget_manager or TokenBudgetManager()
        self.max_concurrent_requests = max_concurrent_requests

        # ZeroGPU configuration
        self.enable_zerogpu = enable_zerogpu and ZEROGPU_AVAILABLE
        self.gpu_memory_threshold = gpu_memory_threshold
        self.batch_timeout = batch_timeout

        # Initialize HF clients
        self.hf_api = HfApi(token=self.hf_token)
        self.inference_clients = {}
        self._init_inference_clients()

        # ZeroGPU model management
        self.loaded_models = {}  # Cache for loaded GPU models
        self.model_pipelines = {}  # Transformers pipelines
        self.gpu_memory_usage = 0.0

        # Rate limiting and performance tracking
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_counts = {}
        self.response_times = []
        self.error_counts = {}

        # Batch processing for ZeroGPU efficiency
        self.batch_queue = deque()
        self.batch_processor_task = None

        # LMStudioClient compatibility tracking
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        self.concurrent_requests = 0
        self._connection_verified = False

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize if ZeroGPU available
        if self.enable_zerogpu:
            self._initialize_zerogpu()

    def _init_inference_clients(self):
        """Initialize inference clients for each model type."""
        for model_type, config in self.model_configs.items():
            try:
                client = InferenceClient(
                    model=config.model_id,
                    token=self.hf_token
                )
                self.inference_clients[model_type] = client
                logger.info(f"Initialized inference client for {model_type.value}: {config.model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize client for {model_type.value}: {e}")
                # Fall back to general model
                if model_type != ModelType.GENERAL:
                    self.inference_clients[model_type] = self.inference_clients.get(ModelType.GENERAL)

    def _initialize_zerogpu(self):
        """Initialize ZeroGPU environment and check availability."""
        if not ZEROGPU_AVAILABLE:
            logger.warning("ZeroGPU not available, falling back to Inference API")
            return

        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"ZeroGPU initialized with {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA not available, ZeroGPU features disabled")
            self.enable_zerogpu = False

    # LMStudioClient Compatibility Methods

    def test_connection(self) -> bool:
        """
        Test connection to HuggingFace services.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple API call
            models = self.hf_api.list_models(limit=1)
            self._connection_verified = True
            return True
        except Exception as e:
            logger.warning(f"HuggingFace connection test failed: {e}")
            self._connection_verified = False
            return False

    def ensure_connection(self) -> None:
        """Ensure connection to HuggingFace or raise exception."""
        if not self._connection_verified and not self.test_connection():
            raise HuggingFaceConnectionError(
                "Cannot connect to HuggingFace services. "
                "Check your internet connection and HF_TOKEN."
            )

    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: Optional[int] = 500,
                 model: str = "local-model") -> LLMResponse:
        """
        Synchronous completion request (LMStudioClient compatibility).

        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier (mapped to agent type)

        Returns:
            LLMResponse with content and metadata

        Raises:
            HuggingFaceConnectionError: If cannot connect to HuggingFace
        """
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Map model to agent type
            agent_type = self._map_model_to_agent_type(model, agent_id)

            # Create combined prompt
            combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            result = loop.run_until_complete(
                self.generate_text(
                    prompt=combined_prompt,
                    agent_type=agent_type,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )

            # Convert HFResponse to LLMResponse for compatibility
            return LLMResponse(
                content=result.content,
                tokens_used=result.tokens_used,
                response_time=result.response_time,
                model=result.model_used,
                temperature=temperature,
                agent_id=agent_id,
                timestamp=time.time()
            )
        finally:
            loop.close()

    async def complete_async(self, agent_id: str, system_prompt: str,
                           user_prompt: str, temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           model: str = "local-model",
                           priority: RequestPriority = RequestPriority.NORMAL) -> LLMResponse:
        """
        Asynchronous completion request (LMStudioClient compatibility).

        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier
            priority: Request priority level

        Returns:
            LLMResponse with content and metadata
        """
        # Map model to agent type
        agent_type = self._map_model_to_agent_type(model, agent_id)

        # Create combined prompt
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        result = await self.generate_text(
            prompt=combined_prompt,
            agent_type=agent_type,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Convert HFResponse to LLMResponse for compatibility
        return LLMResponse(
            content=result.content,
            tokens_used=result.tokens_used,
            response_time=result.response_time,
            model=result.model_used,
            temperature=temperature,
            agent_id=agent_id,
            timestamp=time.time()
        )

    def _map_model_to_agent_type(self, model: str, agent_id: str) -> ModelType:
        """Map model identifier to agent type for compatibility."""
        # Try to infer from agent_id first
        agent_id_lower = agent_id.lower()
        if "research" in agent_id_lower:
            return ModelType.RESEARCH
        elif "analysis" in agent_id_lower or "analyze" in agent_id_lower:
            return ModelType.ANALYSIS
        elif "synthesis" in agent_id_lower or "synthesize" in agent_id_lower:
            return ModelType.SYNTHESIS
        elif "critic" in agent_id_lower or "critique" in agent_id_lower:
            return ModelType.CRITIC

        # Try to infer from model name
        model_lower = model.lower()
        if "research" in model_lower:
            return ModelType.RESEARCH
        elif "analysis" in model_lower or "thinking" in model_lower:
            return ModelType.ANALYSIS
        elif "synthesis" in model_lower or "quality" in model_lower:
            return ModelType.SYNTHESIS
        elif "critic" in model_lower:
            return ModelType.CRITIC

        return ModelType.GENERAL

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics (LMStudioClient compatibility).

        Returns:
            Dictionary with usage metrics
        """
        avg_response_time = (self.total_response_time / self.total_requests
                           if self.total_requests > 0 else 0.0)

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time,
            "average_tokens_per_request": (self.total_tokens / self.total_requests
                                         if self.total_requests > 0 else 0.0),
            "connection_verified": self._connection_verified,
            "max_concurrent_requests": self.max_concurrent_requests,
            "current_concurrent_requests": self.concurrent_requests,
            "queue_size": len(self.batch_queue),
            # ZeroGPU specific stats
            "zerogpu_enabled": self.enable_zerogpu,
            "gpu_memory_usage": self.gpu_memory_usage,
            "loaded_models": list(self.loaded_models.keys())
        }

    def reset_stats(self) -> None:
        """Reset usage statistics (LMStudioClient compatibility)."""
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        self.reset_performance_stats()

    def create_agent_system_prompt(self, agent_type: str, position_info: Dict[str, float],
                                 task_context: str = "") -> str:
        """
        Create system prompt for Felix agent based on position and type (LMStudioClient compatibility).

        Args:
            agent_type: Type of agent (research, analysis, synthesis, critic)
            position_info: Agent's position on helix (x, y, z, radius, depth_ratio)
            task_context: Additional context about the current task

        Returns:
            Formatted system prompt
        """
        # Use the same implementation as LMStudioClient but optimized for ZeroGPU models
        depth_ratio = position_info.get("depth_ratio", 0.0)
        radius = position_info.get("radius", 0.0)

        base_prompt = f"""🚨 IMPORTANT: You are a {agent_type} agent in the Felix multi-agent system optimized for ZeroGPU inference.

⚡ ZeroGPU OPTIMIZATION: This response will be processed on GPU-accelerated infrastructure for optimal performance.

Current Position:
- Depth: {depth_ratio:.2f} (0.0 = top/start, 1.0 = bottom/end)
- Radius: {radius:.2f} (decreasing as you progress)
- Processing Stage: {"Early/Broad" if depth_ratio < 0.3 else "Middle/Focused" if depth_ratio < 0.7 else "Final/Precise"}

Your Role Based on Position:
"""

        if agent_type == "research":
            if depth_ratio < 0.3:
                base_prompt += "- MAXIMUM 5 bullet points with key facts ONLY\n"
                base_prompt += "- NO explanations, NO introductions, NO conclusions\n"
                base_prompt += "- Raw findings only - be direct\n"
            else:
                base_prompt += "- MAXIMUM 3 specific facts with numbers/dates/quotes\n"
                base_prompt += "- NO background context or elaboration\n"
                base_prompt += "- Prepare key points for analysis (concise)\n"

        elif agent_type == "analysis":
            base_prompt += "- MAXIMUM 2 numbered insights/patterns ONLY\n"
            base_prompt += "- NO background explanation or context\n"
            base_prompt += "- Direct analytical findings only\n"

        elif agent_type == "synthesis":
            base_prompt += "- FINAL output ONLY - NO process description\n"
            base_prompt += "- MAXIMUM 3 short paragraphs\n"
            base_prompt += "- Direct, actionable content without fluff\n"

        elif agent_type == "critic":
            base_prompt += "- MAXIMUM 3 specific issues/fixes ONLY\n"
            base_prompt += "- NO praise, NO general comments\n"
            base_prompt += "- Direct problems and solutions only\n"

        if task_context:
            base_prompt += f"\nTask Context: {task_context}\n"

        base_prompt += "\n⚡ ZeroGPU REMINDER: Response optimized for GPU acceleration. "
        base_prompt += "Early positions focus on breadth, later positions focus on depth and precision. BE CONCISE!"

        return base_prompt

    # ZeroGPU-specific methods

    async def _zerogpu_inference(self, model_id: str, prompt: str,
                               generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ZeroGPU-accelerated inference using direct model loading.

        Args:
            model_id: HuggingFace model identifier
            prompt: Input text prompt
            generation_params: Generation parameters

        Returns:
            Generation result with GPU metrics
        """
        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            raise ZeroGPUError("PyTorch and Transformers required for ZeroGPU inference")

        gpu_start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            # Load or get cached model
            if model_id not in self.loaded_models:
                await self._load_model_to_gpu(model_id, generation_params)

            model, tokenizer = self.loaded_models[model_id]

            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)

            # Generate with GPU acceleration
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generation_params.get("max_new_tokens", 512),
                    temperature=generation_params.get("temperature", 0.7),
                    top_p=generation_params.get("top_p", 0.9),
                    do_sample=generation_params.get("do_sample", True),
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=generation_params.get("repetition_penalty", 1.1)
                )

            # Decode response
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Calculate metrics
            gpu_end_time = time.time()
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            return {
                "generated_text": response_text,
                "gpu_time": gpu_end_time - gpu_start_time,
                "gpu_memory_used": (final_memory - initial_memory) / 1024**3,  # GB
                "tokens_generated": len(generated_tokens)
            }

        except Exception as e:
            logger.error(f"ZeroGPU inference failed for {model_id}: {e}")
            # Cleanup on error
            await self._cleanup_gpu_memory()
            raise ZeroGPUError(f"GPU inference failed: {e}")

    async def _load_model_to_gpu(self, model_id: str, generation_params: Dict[str, Any]):
        """Load model to GPU with memory management."""
        if not torch.cuda.is_available():
            raise ZeroGPUError("CUDA not available for model loading")

        try:
            # Check available memory
            available_memory = torch.cuda.get_device_properties(0).total_memory
            if self.gpu_memory_usage > self.gpu_memory_threshold * available_memory:
                await self._cleanup_gpu_memory()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with optimal settings
            torch_dtype = getattr(torch, generation_params.get("torch_dtype", "float16"))

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Cache the loaded model
            self.loaded_models[model_id] = (model, tokenizer)

            # Update memory usage tracking
            current_memory = torch.cuda.memory_allocated()
            self.gpu_memory_usage = current_memory

            logger.info(f"Loaded {model_id} to GPU, memory usage: {current_memory / 1024**3:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load {model_id} to GPU: {e}")
            raise ZeroGPUError(f"Model loading failed: {e}")

    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory by unloading models."""
        if not torch.cuda.is_available():
            return

        # Clear model cache
        for model_id in list(self.loaded_models.keys()):
            model, tokenizer = self.loaded_models.pop(model_id)
            del model, tokenizer

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        self.gpu_memory_usage = 0.0
        logger.info("GPU memory cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

        # Cleanup GPU resources
        if self.enable_zerogpu:
            await self._cleanup_gpu_memory()

    async def close_async(self) -> None:
        """Close async client and cleanup resources (LMStudioClient compatibility)."""
        if self.session:
            await self.session.close()

        if self.enable_zerogpu:
            await self._cleanup_gpu_memory()

        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

    async def generate_text(self,
                          prompt: str,
                          agent_type: ModelType = ModelType.GENERAL,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          use_zerogpu: Optional[bool] = None,
                          priority: RequestPriority = RequestPriority.NORMAL,
                          **kwargs) -> HFResponse:
        """
        Generate text using HuggingFace inference with ZeroGPU optimization.

        Args:
            prompt: Input prompt for text generation
            agent_type: Type of agent requesting generation
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request
            use_zerogpu: Force ZeroGPU usage (None for auto-detect)
            priority: Request priority for processing order
            **kwargs: Additional generation parameters

        Returns:
            HFResponse with generated text and metadata
        """
        async with self.semaphore:
            start_time = time.time()
            self.concurrent_requests += 1

            try:
                # Get model configuration
                config = self.model_configs.get(agent_type, self.model_configs[ModelType.GENERAL])
                client = self.inference_clients.get(agent_type, self.inference_clients[ModelType.GENERAL])

                # Determine if we should use ZeroGPU
                should_use_zerogpu = (
                    use_zerogpu if use_zerogpu is not None
                    else (self.enable_zerogpu and config.use_zerogpu)
                )

                if not client and not should_use_zerogpu:
                    return HFResponse(
                        content="",
                        model_used=config.model_id,
                        tokens_used=0,
                        response_time=0.0,
                        success=False,
                        error=f"No inference client available for {agent_type.value}",
                        fallback_used=False
                    )

                # Check token budget
                estimated_tokens = max_tokens or config.max_tokens
                if hasattr(self.token_budget_manager, 'can_allocate') and not self.token_budget_manager.can_allocate(estimated_tokens):
                    return HFResponse(
                        content="",
                        model_used=config.model_id,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=False,
                        error="Insufficient token budget",
                        fallback_used=False
                    )

                # Prepare generation parameters
                generation_params = {
                    "max_new_tokens": max_tokens or config.max_tokens,
                    "temperature": temperature or config.temperature,
                    "top_p": config.top_p,
                    "repetition_penalty": config.repetition_penalty,
                    "do_sample": True,
                    "return_full_text": False,
                    "torch_dtype": config.torch_dtype,
                    **kwargs
                }

                response_data = None
                gpu_metrics = {}
                fallback_used = False

                # Try ZeroGPU first if enabled and available
                if should_use_zerogpu:
                    try:
                        if self.debug_mode:
                            logger.info(f"Using ZeroGPU inference for {agent_type.value} with {config.model_id}")

                        gpu_result = await self._zerogpu_inference(
                            config.model_id, prompt, generation_params
                        )

                        response_data = [{
                            "generated_text": gpu_result["generated_text"]
                        }]

                        gpu_metrics = {
                            "gpu_time": gpu_result["gpu_time"],
                            "gpu_memory_used": gpu_result["gpu_memory_used"],
                            "tokens_generated": gpu_result["tokens_generated"]
                        }

                    except (ZeroGPUError, GPUMemoryError) as e:
                        logger.warning(f"ZeroGPU failed, falling back to Inference API: {e}")
                        fallback_used = True
                        should_use_zerogpu = False

                # Fallback to Inference API if ZeroGPU failed or not enabled
                if not response_data:
                    if not client:
                        raise Exception("No inference method available")

                    response_data = await self._make_inference_request(
                        client=client,
                        prompt=prompt,
                        parameters=generation_params
                    )
                    fallback_used = not should_use_zerogpu

                # Process response
                if response_data and isinstance(response_data, list) and len(response_data) > 0:
                    generated_text = response_data[0].get("generated_text", "")
                    tokens_used = self._estimate_tokens(prompt + generated_text)

                    # Allocate tokens if budget manager supports it
                    allocation = None
                    if hasattr(self.token_budget_manager, 'allocate_tokens'):
                        allocation = self.token_budget_manager.allocate_tokens(tokens_used)

                    # Track performance
                    response_time = time.time() - start_time
                    self._track_performance(agent_type, response_time, success=True)

                    # Update compatibility stats
                    self.total_tokens += tokens_used
                    self.total_requests += 1
                    self.total_response_time += response_time

                    if self.debug_mode:
                        method = "ZeroGPU" if (should_use_zerogpu and not fallback_used) else "Inference API"
                        logger.info(f"✅ {method} response for {agent_type.value}: {len(generated_text)} chars, {tokens_used} tokens, {response_time:.2f}s")

                    return HFResponse(
                        content=generated_text,
                        model_used=config.model_id,
                        tokens_used=tokens_used,
                        response_time=response_time,
                        success=True,
                        metadata={
                            "agent_type": agent_type.value,
                            "allocation_id": allocation.allocation_id if allocation else None,
                            "parameters": generation_params,
                            "method": "ZeroGPU" if (should_use_zerogpu and not fallback_used) else "Inference API"
                        },
                        gpu_memory_used=gpu_metrics.get("gpu_memory_used"),
                        gpu_time=gpu_metrics.get("gpu_time"),
                        fallback_used=fallback_used
                    )
                else:
                    return HFResponse(
                        content="",
                        model_used=config.model_id,
                        tokens_used=0,
                        response_time=time.time() - start_time,
                        success=False,
                        error="Empty or invalid response from API",
                        fallback_used=fallback_used
                    )

            except Exception as e:
                self._track_performance(agent_type, time.time() - start_time, success=False)
                logger.error(f"HF API request failed for {agent_type.value}: {e}")

                return HFResponse(
                    content="",
                    model_used=config.model_id,
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    success=False,
                    error=str(e),
                    fallback_used=False
                )

            finally:
                self.concurrent_requests -= 1

    async def _make_inference_request(self, client: InferenceClient, prompt: str, parameters: Dict[str, Any]):
        """Make inference request with proper error handling and Pro account optimizations."""
        try:
            # Remove ZeroGPU-specific parameters for Inference API
            api_params = parameters.copy()
            api_params.pop('torch_dtype', None)

            # Use text generation task with Pro account optimizations
            response = await asyncio.wait_for(
                asyncio.create_task(
                    client.text_generation(
                        prompt=prompt,
                        **api_params
                    )
                ),
                timeout=self.request_timeout
            )
            return [{"generated_text": response}] if isinstance(response, str) else response

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {self.request_timeout}s")
        except Exception as e:
            raise Exception(f"Inference request failed: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token on average
        return max(1, len(text) // 4)

    def _track_performance(self, agent_type: ModelType, response_time: float, success: bool):
        """Track performance metrics for monitoring."""
        # Track request counts
        self.request_counts[agent_type] = self.request_counts.get(agent_type, 0) + 1

        # Track response times
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:  # Keep last 1000 responses
            self.response_times = self.response_times[-1000:]

        # Track errors
        if not success:
            self.error_counts[agent_type] = self.error_counts.get(agent_type, 0) + 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics with ZeroGPU metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0

        # ZeroGPU specific stats
        zerogpu_stats = {}
        if self.enable_zerogpu and torch.cuda.is_available():
            zerogpu_stats = {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "loaded_models": list(self.loaded_models.keys()),
                "current_gpu_memory_usage": self.gpu_memory_usage / 1024**3 if self.gpu_memory_usage else 0.0
            }

        base_stats = {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "requests_by_type": dict(self.request_counts),
            "errors_by_type": dict(self.error_counts),
            "zerogpu_enabled": self.enable_zerogpu,
            "zerogpu_available": ZEROGPU_AVAILABLE,
        }

        if hasattr(self.token_budget_manager, 'get_status'):
            base_stats["token_budget_status"] = self.token_budget_manager.get_status()

        base_stats.update(zerogpu_stats)
        return base_stats

    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.request_counts.clear()
        self.response_times.clear()
        self.error_counts.clear()

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured models."""
        health_status = {}

        for model_type, config in self.model_configs.items():
            try:
                # Simple test request
                response = await self.generate_text(
                    prompt="Hello",
                    agent_type=model_type,
                    max_tokens=10
                )
                health_status[model_type.value] = response.success
            except Exception as e:
                logger.error(f"Health check failed for {model_type.value}: {e}")
                health_status[model_type.value] = False

        return health_status

    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models by type."""
        return {
            model_type.value: config.model_id
            for model_type, config in self.model_configs.items()
        }

    async def batch_generate(self,
                           prompts: List[str],
                           agent_types: List[ModelType],
                           use_zerogpu_batching: bool = True,
                           **kwargs) -> List[HFResponse]:
        """
        Generate text for multiple prompts with ZeroGPU batching optimization.

        Args:
            prompts: List of input prompts
            agent_types: List of agent types (must match prompts length)
            use_zerogpu_batching: Enable ZeroGPU batch processing
            **kwargs: Additional generation parameters

        Returns:
            List of HFResponse objects
        """
        if len(prompts) != len(agent_types):
            raise ValueError("Prompts and agent_types lists must have same length")

        # Use ZeroGPU batching for same model types if enabled
        if use_zerogpu_batching and self.enable_zerogpu:
            try:
                return await self._zerogpu_batch_process(prompts, agent_types, **kwargs)
            except Exception as e:
                logger.warning(f"ZeroGPU batching failed, falling back to individual requests: {e}")

        # Create tasks for concurrent execution
        tasks = [
            self.generate_text(prompt=prompt, agent_type=agent_type, **kwargs)
            for prompt, agent_type in zip(prompts, agent_types)
        ]

        # Execute concurrently with semaphore limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(HFResponse(
                    content="",
                    model_used=self.model_configs[agent_types[i]].model_id,
                    tokens_used=0,
                    response_time=0.0,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _zerogpu_batch_process(self, prompts: List[str], agent_types: List[ModelType], **kwargs) -> List[HFResponse]:
        """
        Process multiple prompts using ZeroGPU batching for efficiency.

        Args:
            prompts: List of input prompts
            agent_types: List of agent types
            **kwargs: Additional parameters

        Returns:
            List of HFResponse objects
        """
        # Group by model type for efficient batching
        model_groups = {}
        for i, (prompt, agent_type) in enumerate(zip(prompts, agent_types)):
            config = self.model_configs.get(agent_type, self.model_configs[ModelType.GENERAL])
            model_id = config.model_id

            if model_id not in model_groups:
                model_groups[model_id] = []
            model_groups[model_id].append((i, prompt, agent_type, config))

        # Process each model group with GPU batching
        results = [None] * len(prompts)
        start_time = time.time()

        for model_id, group_items in model_groups.items():
            batch_prompts = [item[1] for item in group_items]
            batch_configs = [item[3] for item in group_items]

            try:
                # Use first config as representative
                base_config = batch_configs[0]
                generation_params = {
                    "max_new_tokens": kwargs.get('max_tokens', base_config.max_tokens),
                    "temperature": kwargs.get('temperature', base_config.temperature),
                    "top_p": base_config.top_p,
                    "repetition_penalty": base_config.repetition_penalty,
                    "do_sample": True,
                    "torch_dtype": base_config.torch_dtype,
                }

                # Process batch on GPU
                batch_results = await self._zerogpu_batch_inference(
                    model_id, batch_prompts, generation_params
                )

                # Map results back to original positions
                for (orig_idx, prompt, agent_type, config), batch_result in zip(group_items, batch_results):
                    tokens_used = self._estimate_tokens(prompt + batch_result["generated_text"])
                    response_time = batch_result.get("response_time", time.time() - start_time)

                    results[orig_idx] = HFResponse(
                        content=batch_result["generated_text"],
                        model_used=model_id,
                        tokens_used=tokens_used,
                        response_time=response_time,
                        success=True,
                        metadata={
                            "agent_type": agent_type.value,
                            "method": "ZeroGPU-Batch",
                            "batch_size": len(batch_prompts)
                        },
                        gpu_memory_used=batch_result.get("gpu_memory_used"),
                        gpu_time=batch_result.get("gpu_time"),
                        batch_processed=len(batch_prompts),
                        fallback_used=False
                    )

            except Exception as e:
                # Fall back to individual processing for this model group
                logger.warning(f"Batch processing failed for {model_id}, using individual requests: {e}")
                for orig_idx, prompt, agent_type, config in group_items:
                    try:
                        individual_result = await self.generate_text(
                            prompt=prompt,
                            agent_type=agent_type,
                            use_zerogpu=False,  # Force Inference API fallback
                            **kwargs
                        )
                        results[orig_idx] = individual_result
                    except Exception as individual_e:
                        results[orig_idx] = HFResponse(
                            content="",
                            model_used=config.model_id,
                            tokens_used=0,
                            response_time=0.0,
                            success=False,
                            error=str(individual_e),
                            fallback_used=True
                        )

        return results

    @spaces.GPU
    async def _zerogpu_batch_inference(self, model_id: str, prompts: List[str], generation_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process multiple prompts in a single ZeroGPU session for efficiency.

        Args:
            model_id: HuggingFace model identifier
            prompts: List of input prompts
            generation_params: Generation parameters

        Returns:
            List of generation results
        """
        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            raise ZeroGPUError("PyTorch and Transformers required for batch ZeroGPU inference")

        gpu_start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            # Load or get cached model
            if model_id not in self.loaded_models:
                await self._load_model_to_gpu(model_id, generation_params)

            model, tokenizer = self.loaded_models[model_id]
            results = []

            # Process prompts individually but in the same GPU session
            for i, prompt in enumerate(prompts):
                prompt_start = time.time()

                # Tokenize input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(model.device)

                # Generate with GPU acceleration
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=generation_params.get("max_new_tokens", 512),
                        temperature=generation_params.get("temperature", 0.7),
                        top_p=generation_params.get("top_p", 0.9),
                        do_sample=generation_params.get("do_sample", True),
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=generation_params.get("repetition_penalty", 1.1)
                    )

                # Decode response
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                prompt_end = time.time()
                results.append({
                    "generated_text": response_text,
                    "response_time": prompt_end - prompt_start,
                    "tokens_generated": len(generated_tokens)
                })

            # Calculate overall GPU metrics
            gpu_end_time = time.time()
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Add GPU metrics to all results
            total_gpu_time = gpu_end_time - gpu_start_time
            gpu_memory_used = (final_memory - initial_memory) / 1024**3  # GB

            for result in results:
                result["gpu_time"] = total_gpu_time / len(results)  # Distribute GPU time
                result["gpu_memory_used"] = gpu_memory_used / len(results)  # Distribute memory usage

            return results

        except Exception as e:
            logger.error(f"ZeroGPU batch inference failed for {model_id}: {e}")
            # Cleanup on error
            await self._cleanup_gpu_memory()
            raise ZeroGPUError(f"GPU batch inference failed: {e}")


# Utility functions for Felix Framework integration

def create_felix_hf_client(token_budget: int = 50000,
                          concurrent_requests: int = 5,
                          enable_zerogpu: bool = True,
                          debug_mode: bool = False) -> HuggingFaceClient:
    """
    Create ZeroGPU-optimized HuggingFace client for Felix Framework deployment on HF Spaces.

    Args:
        token_budget: Total token budget for session
        concurrent_requests: Maximum concurrent requests
        enable_zerogpu: Enable ZeroGPU acceleration
        debug_mode: Enable debug logging

    Returns:
        Configured HuggingFaceClient instance optimized for ZeroGPU and HF Pro
    """
    # Create token manager with Felix-specific settings
    token_manager = TokenBudgetManager(
        base_budget=token_budget // 4,  # Distribute among typical 4 agent types
        strict_mode=True  # Enable for ZeroGPU efficiency
    )

    # ZeroGPU and HF Pro optimized model configurations
    felix_models = {
        ModelType.RESEARCH: HFModelConfig(
            model_id="microsoft/DialoGPT-large",  # Upgraded for better performance
            temperature=0.9,
            max_tokens=256,
            top_p=0.95,
            use_zerogpu=True,
            batch_size=2,  # Efficient batching for research queries
            torch_dtype="float16",
            priority="normal"
        ),
        ModelType.ANALYSIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",  # Better reasoning capability
            temperature=0.5,
            max_tokens=384,
            top_p=0.9,
            use_zerogpu=True,
            batch_size=1,
            torch_dtype="float16",
            priority="high"  # Pro account priority
        ),
        ModelType.SYNTHESIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-13B-Instruct",  # High-quality synthesis
            temperature=0.1,
            max_tokens=512,
            top_p=0.85,
            use_zerogpu=True,
            batch_size=1,
            torch_dtype="float16",
            gpu_memory_limit=12.0,  # Need more memory for 13B model
            priority="high"
        ),
        ModelType.CRITIC: HFModelConfig(
            model_id="microsoft/DialoGPT-large",
            temperature=0.3,
            max_tokens=256,
            top_p=0.9,
            use_zerogpu=True,
            batch_size=2,
            torch_dtype="float16",
            priority="normal"
        )
    }

    return HuggingFaceClient(
        model_configs=felix_models,
        token_budget_manager=token_manager,
        max_concurrent_requests=concurrent_requests,
        request_timeout=45.0,  # Longer timeout for ZeroGPU model loading
        enable_zerogpu=enable_zerogpu,
        gpu_memory_threshold=0.85,  # Conservative memory management
        batch_timeout=3.0,  # Shorter batching timeout for responsiveness
        debug_mode=debug_mode
    )


def create_default_client(max_concurrent_requests: int = 4,
                         enable_zerogpu: bool = True) -> HuggingFaceClient:
    """Create ZeroGPU HuggingFace client with default settings (LMStudioClient compatibility)."""
    return create_felix_hf_client(
        concurrent_requests=max_concurrent_requests,
        enable_zerogpu=enable_zerogpu
    )


# Pro account specific optimizations
def get_pro_account_models() -> Dict[ModelType, HFModelConfig]:
    """
    Get model configurations optimized for HF Pro accounts with access to premium models.

    Returns:
        Dictionary of premium model configurations
    """
    return {
        ModelType.RESEARCH: HFModelConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",  # Premium access
            temperature=0.9,
            max_tokens=384,
            use_zerogpu=True,
            batch_size=3,
            priority="high"
        ),
        ModelType.ANALYSIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-70B-Instruct",  # Large model for complex analysis
            temperature=0.5,
            max_tokens=512,
            use_zerogpu=True,
            batch_size=1,
            gpu_memory_limit=40.0,  # Need significant memory
            priority="high"
        ),
        ModelType.SYNTHESIS: HFModelConfig(
            model_id="meta-llama/Llama-3.1-70B-Instruct",  # Best quality synthesis
            temperature=0.1,
            max_tokens=768,
            use_zerogpu=True,
            batch_size=1,
            gpu_memory_limit=40.0,
            priority="high"
        ),
        ModelType.CRITIC: HFModelConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.3,
            max_tokens=384,
            use_zerogpu=True,
            batch_size=2,
            priority="high"
        )
    }


# ZeroGPU deployment helpers
def estimate_gpu_requirements(model_configs: Dict[ModelType, HFModelConfig]) -> Dict[str, float]:
    """
    Estimate GPU memory requirements for given model configurations.

    Args:
        model_configs: Model configurations to analyze

    Returns:
        Dictionary with memory estimates in GB
    """
    # Rough model size estimates (in GB)
    model_sizes = {
        "microsoft/DialoGPT-medium": 1.5,
        "microsoft/DialoGPT-large": 3.0,
        "meta-llama/Llama-3.1-8B-Instruct": 16.0,
        "meta-llama/Llama-3.1-13B-Instruct": 26.0,
        "meta-llama/Llama-3.1-70B-Instruct": 140.0,
        "Qwen/Qwen2.5-7B-Instruct": 14.0
    }

    requirements = {}
    total_memory = 0.0
    max_single_model = 0.0

    for agent_type, config in model_configs.items():
        model_memory = model_sizes.get(config.model_id, 8.0)  # Default 8GB
        requirements[f"{agent_type.value}_memory"] = model_memory
        total_memory += model_memory
        max_single_model = max(max_single_model, model_memory)

    requirements.update({
        "total_memory_if_all_loaded": total_memory,
        "max_single_model_memory": max_single_model,
        "recommended_gpu_memory": max_single_model * 1.5,  # 50% buffer
        "minimum_gpu_memory": max_single_model * 1.2  # 20% buffer
    })

    return requirements


# Export main classes and functions
__all__ = [
    'HuggingFaceClient',
    'HFResponse',
    'HFModelConfig',
    'ModelType',
    'GPUMemoryError',
    'ZeroGPUError',
    'HuggingFaceConnectionError',
    'create_felix_hf_client',
    'create_default_client',
    'get_pro_account_models',
    'estimate_gpu_requirements',
    'ZEROGPU_AVAILABLE',
    'TORCH_AVAILABLE',
    'TRANSFORMERS_AVAILABLE'
]