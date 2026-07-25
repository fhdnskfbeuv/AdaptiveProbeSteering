"""Steered Mistral for vLLM (``layer_type`` injection; Mistral subclasses Llama)."""

from ._common import SteeringMLP, logger, make_steered_model

try:
	from vllm.model_executor.models.mistral import MistralDecoderLayer
	from vllm.model_executor.models.mistral import MistralForCausalLM as vllmMistral


	class SteeredMistralDecoderLayer(MistralDecoderLayer):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			# Instance attribute (NOT a class attribute); None = layer not steered.
			self.steer_mlp: SteeringMLP | None = None

		def forward(self, positions, hidden_states, residual):
			if self.steer_mlp is not None:
				stream = hidden_states if residual is None else hidden_states + residual
				hidden_states = hidden_states + self.steer_mlp(stream)
			return super().forward(positions, hidden_states, residual)


	vllmSteeredMistralForCausalLM = make_steered_model(
		vllmMistral, SteeredMistralDecoderLayer, use_layer_type=True,
	)

except ImportError as e:
	logger.warning('Steered Mistral unavailable: %s', e)
