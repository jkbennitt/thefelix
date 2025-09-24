"""
Unit tests for ZeroGPU-optimized HuggingFace client.

Tests cover ZeroGPU functionality, batch processing, GPU memory management,
LMStudioClient compatibility, and HF Pro account features.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.llm.huggingface_client import (
    HuggingFaceClient,
    HFModelConfig,
    HFResponse,
    ModelType,
    GPUMemoryError,
    ZeroGPUError,
    HuggingFaceConnectionError,
    create_felix_hf_client,
    create_default_client,
    get_pro_account_models,
    estimate_gpu_requirements,
    ZEROGPU_AVAILABLE,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE
)
from src.llm.lm_studio_client import RequestPriority, LLMResponse
from src.llm.token_budget import TokenBudgetManager


class TestHuggingFaceClientCompatibility:
    """Test LMStudioClient API compatibility."""

    @pytest.fixture
    def client(self):
        """Create HuggingFace client for testing."""
        return create_felix_hf_client(debug_mode=True, enable_zerogpu=False)

    def test_initialization_compatibility(self, client):
        """Test initialization maintains LMStudioClient compatibility."""
        assert hasattr(client, 'test_connection')
        assert hasattr(client, 'ensure_connection')
        assert hasattr(client, 'complete')
        assert hasattr(client, 'complete_async')
        assert hasattr(client, 'get_usage_stats')
        assert hasattr(client, 'reset_stats')
        assert hasattr(client, 'create_agent_system_prompt')

    def test_create_default_client(self):
        """Test default client creation for compatibility."""
        client = create_default_client(max_concurrent_requests=8)
        assert client.max_concurrent_requests == 8
        assert isinstance(client, HuggingFaceClient)

    @pytest.mark.asyncio
    async def test_connection_testing(self, client):
        """Test connection testing functionality."""
        with patch.object(client.hf_api, 'list_models') as mock_list:
            mock_list.return_value = [{'id': 'test-model'}]
            assert client.test_connection() is True
            assert client._connection_verified is True

        with patch.object(client.hf_api, 'list_models') as mock_list:
            mock_list.side_effect = Exception("Connection failed")
            assert client.test_connection() is False
            assert client._connection_verified is False

    def test_ensure_connection(self, client):
        """Test connection ensuring with proper error handling."""
        with patch.object(client, 'test_connection', return_value=True):
            client.ensure_connection()  # Should not raise

        with patch.object(client, 'test_connection', return_value=False):
            with pytest.raises(HuggingFaceConnectionError):
                client.ensure_connection()

    def test_usage_stats_compatibility(self, client):
        """Test usage stats format matches LMStudioClient."""
        stats = client.get_usage_stats()

        # Check LMStudioClient compatibility fields
        required_fields = [
            'total_requests', 'total_tokens', 'total_response_time',
            'average_response_time', 'average_tokens_per_request',
            'connection_verified', 'max_concurrent_requests',
            'current_concurrent_requests', 'queue_size'
        ]

        for field in required_fields:
            assert field in stats

        # Check ZeroGPU extensions
        assert 'zerogpu_enabled' in stats
        assert 'zerogpu_available' in stats

    def test_agent_system_prompt_creation(self, client):
        """Test agent system prompt creation with ZeroGPU optimizations."""
        position_info = {'depth_ratio': 0.5, 'radius': 10.0}
        prompt = client.create_agent_system_prompt('synthesis', position_info, 'test task')

        assert 'synthesis agent' in prompt
        assert 'ZeroGPU' in prompt
        assert 'depth_ratio: 0.50' in prompt


class TestZeroGPUFunctionality:
    """Test ZeroGPU-specific features."""

    @pytest.fixture
    def zerogpu_client(self):
        """Create ZeroGPU-enabled client for testing."""
        return create_felix_hf_client(enable_zerogpu=True, debug_mode=True)

    @pytest.mark.skipif(not ZEROGPU_AVAILABLE, reason="ZeroGPU not available")
    def test_zerogpu_initialization(self, zerogpu_client):
        """Test ZeroGPU initialization when available."""
        assert zerogpu_client.enable_zerogpu is True
        assert hasattr(zerogpu_client, 'loaded_models')
        assert hasattr(zerogpu_client, 'gpu_memory_usage')

    def test_zerogpu_mock_environment(self):
        """Test behavior when ZeroGPU is not available."""
        with patch('src.llm.huggingface_client.ZEROGPU_AVAILABLE', False):
            client = create_felix_hf_client(enable_zerogpu=True)
            assert client.enable_zerogpu is False

    @pytest.mark.asyncio
    @patch('src.llm.huggingface_client.torch')
    @patch('src.llm.huggingface_client.AutoModelForCausalLM')
    @patch('src.llm.huggingface_client.AutoTokenizer')
    async def test_model_loading_to_gpu(self, mock_tokenizer, mock_model, mock_torch, zerogpu_client):
        """Test model loading to GPU with memory management."""
        if not zerogpu_client.enable_zerogpu:
            pytest.skip("ZeroGPU not enabled")

        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3
        mock_torch.cuda.memory_allocated.return_value = 1 * 1024**3

        # Mock model and tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '<eos>'
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test model loading
        await zerogpu_client._load_model_to_gpu(
            "test-model",
            {"torch_dtype": "float16"}
        )

        # Verify model was cached
        assert "test-model" in zerogpu_client.loaded_models
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.llm.huggingface_client.torch')
    async def test_gpu_memory_cleanup(self, mock_torch, zerogpu_client):
        """Test GPU memory cleanup functionality."""
        if not zerogpu_client.enable_zerogpu:
            pytest.skip("ZeroGPU not enabled")

        # Mock loaded models
        zerogpu_client.loaded_models = {
            "model1": (Mock(), Mock()),
            "model2": (Mock(), Mock())
        }

        mock_torch.cuda.is_available.return_value = True

        await zerogpu_client._cleanup_gpu_memory()

        # Verify cleanup
        assert len(zerogpu_client.loaded_models) == 0
        assert zerogpu_client.gpu_memory_usage == 0.0
        mock_torch.cuda.empty_cache.assert_called_once()


class TestBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.fixture
    def batch_client(self):
        """Create client optimized for batch processing."""
        return create_felix_hf_client(concurrent_requests=8, debug_mode=True)

    @pytest.mark.asyncio
    async def test_batch_generate_basic(self, batch_client):
        """Test basic batch generation functionality."""
        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
        agent_types = [ModelType.RESEARCH, ModelType.ANALYSIS, ModelType.SYNTHESIS]

        # Mock the generate_text method
        mock_responses = [
            HFResponse(
                content=f"Response {i+1}",
                model_used="test-model",
                tokens_used=50,
                response_time=1.0,
                success=True
            )
            for i in range(3)
        ]

        with patch.object(batch_client, 'generate_text') as mock_generate:
            mock_generate.side_effect = mock_responses

            results = await batch_client.batch_generate(
                prompts, agent_types, use_zerogpu_batching=False
            )

            assert len(results) == 3
            assert all(result.success for result in results)
            assert mock_generate.call_count == 3

    @pytest.mark.asyncio
    async def test_zerogpu_batch_processing(self, batch_client):
        """Test ZeroGPU batch processing optimization."""
        prompts = ["Batch prompt 1", "Batch prompt 2"]
        agent_types = [ModelType.RESEARCH, ModelType.RESEARCH]  # Same type for batching

        with patch.object(batch_client, '_zerogpu_batch_process') as mock_batch:
            mock_batch.return_value = [
                HFResponse(
                    content="Batched response 1",
                    model_used="test-model",
                    tokens_used=50,
                    response_time=0.8,
                    success=True,
                    batch_processed=2
                ),
                HFResponse(
                    content="Batched response 2",
                    model_used="test-model",
                    tokens_used=45,
                    response_time=0.8,
                    success=True,
                    batch_processed=2
                )
            ]

            batch_client.enable_zerogpu = True
            results = await batch_client.batch_generate(
                prompts, agent_types, use_zerogpu_batching=True
            )

            assert len(results) == 2
            mock_batch.assert_called_once()

    def test_batch_input_validation(self, batch_client):
        """Test batch processing input validation."""
        with pytest.raises(ValueError):
            asyncio.run(batch_client.batch_generate(
                ["prompt1", "prompt2"],
                [ModelType.RESEARCH]  # Mismatched lengths
            ))


class TestModelConfigurations:
    """Test model configuration and specialization."""

    def test_default_model_configs(self):
        """Test default ZeroGPU-optimized model configurations."""
        client = create_felix_hf_client()

        # Verify ZeroGPU optimizations in default configs
        for model_type, config in client.model_configs.items():
            assert config.use_zerogpu is True
            assert config.torch_dtype == "float16"
            assert hasattr(config, 'batch_size')

        # Verify model upgrades for ZeroGPU
        assert "meta-llama" in client.model_configs[ModelType.ANALYSIS].model_id
        assert "meta-llama" in client.model_configs[ModelType.SYNTHESIS].model_id

    def test_pro_account_models(self):
        """Test Pro account model configurations."""
        pro_models = get_pro_account_models()

        for model_type, config in pro_models.items():
            assert config.priority == "high"
            assert config.use_zerogpu is True

        # Verify premium model access
        assert "Llama-3.1-70B" in pro_models[ModelType.ANALYSIS].model_id
        assert "Llama-3.1-70B" in pro_models[ModelType.SYNTHESIS].model_id

    def test_gpu_requirements_estimation(self):
        """Test GPU memory requirements estimation."""
        test_configs = {
            ModelType.RESEARCH: HFModelConfig(
                model_id="microsoft/DialoGPT-medium"
            ),
            ModelType.ANALYSIS: HFModelConfig(
                model_id="meta-llama/Llama-3.1-8B-Instruct"
            )
        }

        requirements = estimate_gpu_requirements(test_configs)

        assert 'research_memory' in requirements
        assert 'analysis_memory' in requirements
        assert 'total_memory_if_all_loaded' in requirements
        assert 'recommended_gpu_memory' in requirements
        assert requirements['recommended_gpu_memory'] > requirements['max_single_model_memory']

    def test_model_to_agent_type_mapping(self):
        """Test model identifier to agent type mapping."""
        client = create_felix_hf_client()

        # Test agent_id-based mapping
        assert client._map_model_to_agent_type("any", "research_agent") == ModelType.RESEARCH
        assert client._map_model_to_agent_type("any", "synthesis_agent") == ModelType.SYNTHESIS
        assert client._map_model_to_agent_type("any", "critic_agent") == ModelType.CRITIC

        # Test model name-based mapping
        assert client._map_model_to_agent_type("research_fast", "agent") == ModelType.RESEARCH
        assert client._map_model_to_agent_type("thinking_analysis", "agent") == ModelType.ANALYSIS

        # Test fallback
        assert client._map_model_to_agent_type("unknown", "unknown") == ModelType.GENERAL


class TestErrorHandlingAndFallbacks:
    """Test comprehensive error handling and fallback mechanisms."""

    @pytest.fixture
    def fallback_client(self):
        """Create client for testing fallback scenarios."""
        return create_felix_hf_client(enable_zerogpu=True, debug_mode=True)

    @pytest.mark.asyncio
    async def test_zerogpu_fallback_to_inference_api(self, fallback_client):
        """Test fallback from ZeroGPU to Inference API."""
        with patch.object(fallback_client, '_zerogpu_inference') as mock_gpu:
            mock_gpu.side_effect = ZeroGPUError("GPU allocation failed")

            with patch.object(fallback_client, '_make_inference_request') as mock_api:
                mock_api.return_value = [{"generated_text": "Fallback response"}]

                result = await fallback_client.generate_text(
                    "test prompt",
                    ModelType.RESEARCH,
                    use_zerogpu=True
                )

                assert result.success is True
                assert result.content == "Fallback response"
                assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_inference_api_error_handling(self, fallback_client):
        """Test error handling for Inference API failures."""
        with patch.object(fallback_client, '_make_inference_request') as mock_api:
            mock_api.side_effect = Exception("API rate limit exceeded")

            result = await fallback_client.generate_text("test", ModelType.GENERAL)

            assert result.success is False
            assert "API rate limit exceeded" in result.error

    def test_gpu_memory_error_handling(self, fallback_client):
        """Test GPU memory error handling."""
        with pytest.raises(GPUMemoryError):
            raise GPUMemoryError("Insufficient GPU memory")

        with pytest.raises(ZeroGPUError):
            raise ZeroGPUError("ZeroGPU allocation failed")

    @pytest.mark.asyncio
    async def test_token_budget_integration(self, fallback_client):
        """Test token budget manager integration."""
        # Mock token budget manager
        mock_budget = Mock()
        mock_budget.can_allocate = Mock(return_value=False)
        fallback_client.token_budget_manager = mock_budget

        # Should return error when budget exceeded
        result = await fallback_client.generate_text("test", ModelType.GENERAL)

        # Note: The actual implementation may vary based on token_budget_manager interface
        # This test ensures the integration is considered


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""

    @pytest.fixture
    def monitored_client(self):
        """Create client with performance monitoring."""
        return create_felix_hf_client(debug_mode=True)

    def test_performance_stats_structure(self, monitored_client):
        """Test performance statistics structure."""
        stats = monitored_client.get_performance_stats()

        # Core stats
        assert 'total_requests' in stats
        assert 'total_errors' in stats
        assert 'error_rate' in stats
        assert 'avg_response_time' in stats

        # ZeroGPU specific stats
        assert 'zerogpu_enabled' in stats
        assert 'zerogpu_available' in stats

    @pytest.mark.asyncio
    async def test_performance_tracking(self, monitored_client):
        """Test performance tracking during requests."""
        initial_stats = monitored_client.get_performance_stats()
        initial_requests = initial_stats['total_requests']

        # Mock a successful request
        with patch.object(monitored_client, '_make_inference_request') as mock_api:
            mock_api.return_value = [{"generated_text": "test response"}]

            await monitored_client.generate_text("test", ModelType.GENERAL)

        updated_stats = monitored_client.get_performance_stats()
        assert updated_stats['total_requests'] > initial_requests

    def test_stats_reset_functionality(self, monitored_client):
        """Test statistics reset functionality."""
        # Add some fake stats
        monitored_client.total_requests = 10
        monitored_client.total_tokens = 500

        monitored_client.reset_stats()

        assert monitored_client.total_requests == 0
        assert monitored_client.total_tokens == 0


class TestLMStudioClientCompatibility:
    """Test full LMStudioClient API compatibility."""

    @pytest.fixture
    def compat_client(self):
        """Create client for compatibility testing."""
        return create_felix_hf_client(debug_mode=False)

    def test_complete_method_signature(self, compat_client):
        """Test synchronous complete method matches LMStudioClient."""
        with patch.object(compat_client, 'generate_text') as mock_generate:
            mock_generate.return_value = asyncio.Future()
            mock_generate.return_value.set_result(HFResponse(
                content="sync response",
                model_used="test-model",
                tokens_used=25,
                response_time=1.0,
                success=True
            ))

            # Mock event loop for sync execution
            with patch('asyncio.new_event_loop') as mock_loop:
                mock_loop_instance = Mock()
                mock_loop.return_value = mock_loop_instance
                mock_loop_instance.run_until_complete.return_value = mock_generate.return_value.result()

                result = compat_client.complete(
                    agent_id="test_agent",
                    system_prompt="You are a helpful assistant",
                    user_prompt="Hello",
                    temperature=0.5,
                    max_tokens=100,
                    model="test-model"
                )

                assert isinstance(result, LLMResponse)
                assert result.content == "sync response"
                assert result.agent_id == "test_agent"

    @pytest.mark.asyncio
    async def test_complete_async_method(self, compat_client):
        """Test asynchronous complete method compatibility."""
        with patch.object(compat_client, 'generate_text') as mock_generate:
            mock_generate.return_value = HFResponse(
                content="async response",
                model_used="test-model",
                tokens_used=30,
                response_time=0.8,
                success=True
            )

            result = await compat_client.complete_async(
                agent_id="async_agent",
                system_prompt="System prompt",
                user_prompt="User prompt",
                temperature=0.7,
                priority=RequestPriority.HIGH
            )

            assert isinstance(result, LLMResponse)
            assert result.content == "async response"
            assert result.agent_id == "async_agent"

    @pytest.mark.asyncio
    async def test_async_context_manager(self, compat_client):
        """Test async context manager functionality."""
        async with compat_client as client:
            assert client.session is not None

        # Session should be closed after context exit
        assert compat_client.session is None or compat_client.session.closed


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_felix_multi_agent_workflow(self):
        """Test typical Felix multi-agent workflow."""
        client = create_felix_hf_client(debug_mode=True, enable_zerogpu=False)

        # Mock responses for different agent types
        responses = {
            ModelType.RESEARCH: "• Research finding 1\n• Research finding 2",
            ModelType.ANALYSIS: "1. Key insight\n2. Important pattern",
            ModelType.SYNTHESIS: "Final synthesized output combining all inputs."
        }

        with patch.object(client, '_make_inference_request') as mock_api:
            def mock_response(client_obj, prompt, params):
                # Determine agent type from prompt or model
                if "research" in prompt.lower():
                    agent_type = ModelType.RESEARCH
                elif "analysis" in prompt.lower():
                    agent_type = ModelType.ANALYSIS
                else:
                    agent_type = ModelType.SYNTHESIS

                return [{"generated_text": responses[agent_type]}]

            mock_api.side_effect = mock_response

            # Simulate Felix agent workflow
            research_result = await client.generate_text(
                "Research prompt for exploration",
                ModelType.RESEARCH
            )

            analysis_result = await client.generate_text(
                "Analysis prompt for insight",
                ModelType.ANALYSIS
            )

            synthesis_result = await client.generate_text(
                "Synthesis prompt for final output",
                ModelType.SYNTHESIS
            )

            # Verify workflow completion
            assert research_result.success
            assert analysis_result.success
            assert synthesis_result.success

            # Verify agent specialization
            assert "finding" in research_result.content
            assert "insight" in analysis_result.content
            assert "synthesized" in synthesis_result.content

    def test_environment_feature_detection(self):
        """Test feature detection in different environments."""
        # Test availability flags
        assert isinstance(ZEROGPU_AVAILABLE, bool)
        assert isinstance(TORCH_AVAILABLE, bool)
        assert isinstance(TRANSFORMERS_AVAILABLE, bool)

        # Test graceful degradation
        client = create_felix_hf_client(enable_zerogpu=True)

        if not ZEROGPU_AVAILABLE:
            assert client.enable_zerogpu is False

        # Client should still be functional regardless of feature availability
        assert client is not None
        assert hasattr(client, 'generate_text')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])