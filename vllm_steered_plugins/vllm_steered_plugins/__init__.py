"""vLLM v1 plugin that registers the probe-steered models.

vLLM v1 only picks up custom architectures through its plugin system. Installing
this package exposes an entrypoint under the ``vllm.general_plugins`` group; vLLM
calls ``register_models()`` at startup, which lazily registers each steered model
class so a checkpoint whose ``config.architectures`` matches one of them loads
correctly.
"""

# arch name (written into config.architectures by transferSteer2Safetensors.py)
# -> "module:class" lazily imported only when the model is instantiated.
_STEERED_MODELS = {
	"vllmSteeredLlamaForCausalLM":
		"vllm_steered_plugins.steered_llama:vllmSteeredLlamaForCausalLM",
	"vllmSteeredMistralForCausalLM":
		"vllm_steered_plugins.steered_mistral:vllmSteeredMistralForCausalLM",
	"vllmSteeredGemma2ForCausalLM":
		"vllm_steered_plugins.steered_gemma2:vllmSteeredGemma2ForCausalLM",
	"vllmSteeredGemma4UnifiedForConditionalGeneration":
		"vllm_steered_plugins.steered_gemma4:vllmSteeredGemma4UnifiedForConditionalGeneration",
	"vllmSteeredQwen3_5MoeForConditionalGeneration":
		"vllm_steered_plugins.steered_qwen3_5:vllmSteeredQwen3_5MoeForConditionalGeneration",
	"vllmSteeredQwen3_5ForConditionalGeneration":
		"vllm_steered_plugins.steered_qwen3_5:vllmSteeredQwen3_5ForConditionalGeneration",
}


def register_models():
	from vllm import ModelRegistry

	# String form => the module is imported lazily, only when the model is
	# actually instantiated (avoids initializing CUDA at registration time).
	for arch, target in _STEERED_MODELS.items():
		ModelRegistry.register_model(arch, target)
