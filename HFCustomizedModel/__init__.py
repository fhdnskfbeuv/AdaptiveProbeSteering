"""Transformers (HF) models that load the probe-steered checkpoints produced by
``transferSteer2Safetensors.py``.

Each ``Steered*`` class subclasses the matching HF model and, in ``__init__``,
recreates the baked ``steer_mlp`` submodule on every layer in
``config.steered_layers`` (so ``from_pretrained`` loads its weights) and runs the
steering through a forward pre-hook. Load directly, e.g.::

    from HFCustomizedModel import SteeredLlamaForCausalLM
    model = SteeredLlamaForCausalLM.from_pretrained(steered_dir)

Imports are guarded so a model class absent from the installed transformers
version (e.g. a newer multimodal arch) does not break the rest of the package.
"""

__all__ = []


def _export(module, names):
	import importlib
	try:
		mod = importlib.import_module(f'{__name__}.{module}')
	except ImportError as e:  # transformers too old for this arch
		import logging
		logging.getLogger('HFCustomizedModel').warning('%s unavailable: %s', module, e)
		return
	for name in names:
		globals()[name] = getattr(mod, name)
		__all__.append(name)


_export('SteeredLlamaForCausalLM', ['SteeredLlamaForCausalLM'])
_export('SteeredMistralForCausalLM', ['SteeredMistralForCausalLM'])
_export('SteeredGemma2ForCausalLM', ['SteeredGemma2ForCausalLM'])
_export('SteeredGemma4UnifiedForConditionalGeneration', ['SteeredGemma4UnifiedForConditionalGeneration'])
_export('SteeredQwen3_5ForConditionalGeneration', [
	'SteeredQwen3_5ForConditionalGeneration',
	'SteeredQwen3_5MoeForConditionalGeneration',
])
