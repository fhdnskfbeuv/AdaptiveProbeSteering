"""Steered Qwen3.5 (dense and MoE, multimodal) for vLLM (instance-replacement injection).

The decoder-layer forward argument order is ``(hidden_states, residual,
positions=None, **kwargs)`` (unlike Llama's ``(positions, hidden_states,
residual)``); Qwen3.5 shares Llama's residual convention, so the inter-layer
stream is ``hidden_states + residual``.
"""

from ._common import logger, make_steered_model

try:
	from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
	from vllm.model_executor.models.qwen3_5 import (
		Qwen3_5ForConditionalGeneration as vllmQwen3_5,
		Qwen3_5MoeForConditionalGeneration as vllmQwen3_5Moe,
	)


	class SteeredQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
		def forward(self, hidden_states, residual, positions=None, **kwargs):
			if self.steer_mlp is not None:
				stream = hidden_states if residual is None else hidden_states + residual
				hidden_states = hidden_states + self.steer_mlp(stream)
			return super().forward(hidden_states, residual, positions=positions, **kwargs)


	vllmSteeredQwen3_5ForConditionalGeneration = make_steered_model(
		vllmQwen3_5, SteeredQwen3_5DecoderLayer, prefix='model',
	)
	vllmSteeredQwen3_5MoeForConditionalGeneration = make_steered_model(
		vllmQwen3_5Moe, SteeredQwen3_5DecoderLayer, prefix='model',
	)

except ImportError as e:
	logger.warning('Steered Qwen3.5 unavailable: %s', e)
