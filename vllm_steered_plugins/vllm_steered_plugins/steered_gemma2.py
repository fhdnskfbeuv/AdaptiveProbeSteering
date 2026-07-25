"""Steered Gemma2 for vLLM (instance-replacement injection).

Gemma2 shares Llama's residual convention: the inter-layer stream is
``hidden_states + residual``.
"""

from ._common import logger, make_steered_model

try:
	from vllm.model_executor.models.gemma2 import Gemma2DecoderLayer
	from vllm.model_executor.models.gemma2 import Gemma2ForCausalLM as vllmGemma2


	class SteeredGemma2DecoderLayer(Gemma2DecoderLayer):
		def forward(self, positions, hidden_states, residual):
			if self.steer_mlp is not None:
				stream = hidden_states if residual is None else hidden_states + residual
				hidden_states = hidden_states + self.steer_mlp(stream)
			return super().forward(positions, hidden_states, residual)


	vllmSteeredGemma2ForCausalLM = make_steered_model(vllmGemma2, SteeredGemma2DecoderLayer)

except ImportError as e:
	logger.warning('Steered Gemma2 unavailable: %s', e)
