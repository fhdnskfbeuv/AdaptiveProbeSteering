"""Steered Llama for vLLM (``layer_type`` injection)."""

from ._common import SteeringMLP, logger, make_steered_model

try:
	from vllm.model_executor.models.llama import LlamaDecoderLayer
	from vllm.model_executor.models.llama import LlamaForCausalLM as vllmLlama


	class SteeredLlamaDecoderLayer(LlamaDecoderLayer):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			# Instance attribute (NOT a class attribute) so torch.compile resolves the
			# real per-layer value; None marks a layer that is not steered.
			self.steer_mlp: SteeringMLP | None = None

		def forward(self, positions, hidden_states, residual):
			if self.steer_mlp is not None:
				stream = hidden_states if residual is None else hidden_states + residual
				hidden_states = hidden_states + self.steer_mlp(stream)
			return super().forward(positions, hidden_states, residual)


	vllmSteeredLlamaForCausalLM = make_steered_model(
		vllmLlama, SteeredLlamaDecoderLayer, use_layer_type=True,
	)

except ImportError as e:
	logger.warning('Steered Llama unavailable: %s', e)
