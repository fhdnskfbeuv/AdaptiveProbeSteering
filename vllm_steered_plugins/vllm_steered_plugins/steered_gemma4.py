"""Steered Gemma4 Unified (multimodal) for vLLM (instance-replacement injection).

Gemma4 resets ``residual = hidden_states`` at the top of each layer, so the
inter-layer stream is ``hidden_states`` alone; the steering is evaluated on and
added to ``hidden_states``.
"""

from ._common import logger, make_steered_model

try:
	from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
	from vllm.model_executor.models.gemma4_unified import (
		Gemma4UnifiedForConditionalGeneration as vllmGemma4Unified,
	)


	class SteeredGemma4DecoderLayer(Gemma4DecoderLayer):
		def forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
			if self.steer_mlp is not None:
				hidden_states = hidden_states + self.steer_mlp(hidden_states)
			return super().forward(positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs)


	vllmSteeredGemma4UnifiedForConditionalGeneration = make_steered_model(
		vllmGemma4Unified, SteeredGemma4DecoderLayer,
	)

except ImportError as e:
	logger.warning('Steered Gemma4 Unified unavailable: %s', e)
