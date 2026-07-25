"""Shared helpers for the steered model modules.

Everything here is pure torch (no vLLM import) so it can also be used by the
offline transfer script. ``SteeringMLP``, the probe helpers and the model/layer
injection helpers live here because every family module reuses them; each family
module only defines its ``Steered*DecoderLayer`` (the family-specific ``forward``)
and builds its model class with :func:`make_steered_model`.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger('vllm_steered_plugins')


# --------------------------------------------------------------------------- #
# Steering as a two-layer MLP (hidden size 1, ReLU)
# --------------------------------------------------------------------------- #
# The probe steering applied to the residual stream x is:
#
#     delta = (relu(s - x @ w.T - b) / wNorm) @ (w / wNorm)
#           = relu(s - x @ w.T - b) * w / wNorm**2
#
# which is exactly a residual MLP with a 1-dim hidden layer and ReLU:
#
#     Linear(D -> 1):  weight = -w, bias = (s - b)   =>  s - x @ w.T - b
#     ReLU
#     Linear(1 -> D):  weight = (w / wNorm**2).T, no bias
#
# so that steered x = x + MLP(x). Baking it into a real nn.Module lets vLLM
# load it straight from the checkpoint instead of a slow forward hook.

class SteeringMLP(nn.Module):
	def __init__(self, hiddenSize: int, dtype: torch.dtype | None = None):
		super().__init__()
		self.down = nn.Linear(hiddenSize, 1, bias=True, dtype=dtype)
		self.up = nn.Linear(1, hiddenSize, bias=False, dtype=dtype)

	@classmethod
	def from_probe(cls, w: torch.Tensor, b: torch.Tensor, s: torch.Tensor, dtype: torch.dtype | None = None):
		# w: [1, D], b: [1], s: [1]. Compute in fp32 for precision, store in `dtype`
		# (typically the base model's dtype so the adapter matches the checkpoint).
		hiddenSize = w.shape[-1]
		w = w.float()
		b = b.float()
		s = s.float()
		wNorm = torch.norm(w, dim=-1)
		wNorm = torch.where(wNorm == 0.0, torch.tensor(1e-6), wNorm)
		module = cls(hiddenSize, dtype=dtype)
		with torch.no_grad():
			module.down.weight.copy_(-w)                   # [1, D]
			module.down.bias.copy_(s - b)                  # [1]
			module.up.weight.copy_((w / (wNorm ** 2)).T)   # [D, 1]
		return module

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# The module is created in the model's compute dtype (see make_steered_model),
		# so its weights already match x.dtype; plain nn.Linear calls are correct and
		# compile-friendly.
		return self.up(torch.relu(self.down(x)))


# --------------------------------------------------------------------------- #
# Probe helpers (loading trained probes and computing target scores)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def getTargetScores(pm, strength: str):
	ret = {}
	for layerIdx, probe in pm.items():
		if strength == 'mean':
			ret[layerIdx] = torch.mean(probe['score'])
		else:
			ret[layerIdx] = torch.quantile(probe['score'], float(strength)) if 'abs' not in strength else torch.tensor(float(strength.replace('abs', '')))
	return ret


def getProbe(allProbes: dict, which):
	if which == 'all':
		probe2test = {k: allProbes[k] for k in sorted(allProbes)}
	elif which == 'first':
		probe2test = {k: allProbes[k] for k in sorted(allProbes)[:1]}
	elif which == 'best':
		probe2test = {k: allProbes[k] for k in sorted(allProbes, key=lambda k: allProbes[k][0], reverse=True)[:1]}
	elif which == 'last':
		probe2test = {k: allProbes[k] for k in sorted(allProbes, reverse=True)[:1]}
	else:
		probe2test = None
	(iterNum, stuff) = list(probe2test.items())[0]
	return stuff[1], f'Iter{iterNum}, {stuff[0]}'


# --------------------------------------------------------------------------- #
# Layer access / injection helpers
# --------------------------------------------------------------------------- #

def get_layers(model):
	"""Return the decoder-layer ModuleList for a vLLM model.

	Handles both plain text models (``model.model.layers``) and multimodal
	wrappers (``model.language_model.model.layers``).
	"""
	base = model
	if hasattr(base, 'language_model'):
		base = base.language_model
	if hasattr(base, 'model'):
		base = base.model
	return base.layers


def hidden_size_of(config) -> int:
	"""Resolve the text hidden size from a (possibly multimodal) HF config."""
	hiddenSize = getattr(config, 'hidden_size', None)
	if hiddenSize is None:
		textConfig = getattr(config, 'text_config', None)
		if textConfig is not None:
			hiddenSize = textConfig.hidden_size
	return hiddenSize


def steered_layers_of(config) -> list:
	"""Layer indices recorded in the merged checkpoint's config."""
	return list(getattr(config, 'steered_layers', None) or [])


def attach_steering(model, steeredLayers, hiddenSize: int, steeredLayerCls, dtype=None):
	"""Give each steered layer a ``SteeringMLP`` and make it run ``steeredLayerCls.forward``.

	Two cases, handled uniformly:

	* ``layer_type`` models (Llama/Mistral) already built the layer as
	  ``steeredLayerCls`` -- we only set ``steer_mlp``.
	* Models that build their own layers (Gemma2/Gemma4/Qwen3.5) get the layer
	  REPLACED by a genuine ``steeredLayerCls`` instance that reuses the
	  already-initialized submodules/weights (we must not rerun ``__init__``, which
	  would re-initialize them).

	Replacing the instance (instead of mutating ``layer.__class__``) keeps a real
	``steeredLayerCls`` object in the ModuleList, which torch.compile dispatches on
	reliably; a runtime ``__class__`` swap is silently ignored by the compiled graph.
	"""
	layers = get_layers(model)
	numLayers = len(layers)
	for target in steeredLayers:
		if target >= numLayers:
			logger.warning('Skip steered layer %d: out of range (%d layers).', target, numLayers)
			continue
		layer = layers[target]
		if not isinstance(layer, steeredLayerCls):
			new = steeredLayerCls.__new__(steeredLayerCls)  # no __init__: keep trained weights
			new.__dict__.update(layer.__dict__)             # inherit submodules/params/buffers
			layers[target] = new
			layer = new
		layer.steer_mlp = SteeringMLP(hiddenSize, dtype=dtype)
		logger.info('Steering layer %d (%s)', target, type(layer).__name__)


def make_steered_model(baseCls, steeredLayerCls, prefix: str = '', use_layer_type: bool = False):
	"""Build a steered vLLM model class wrapping ``baseCls``.

	The generated ``__init__`` defers to ``baseCls`` (passing ``layer_type`` when the
	base supports it) and then activates steering on the layers listed in the
	checkpoint's ``config.steered_layers``. The returned class is named
	``vllmSteered<baseCls.__name__>`` to match the plugin registration.
	"""
	if use_layer_type:
		class SteeredModel(baseCls):
			def __init__(self, *, vllm_config, prefix: str = prefix, layer_type=steeredLayerCls):
				super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
				_activate_steering(self, vllm_config, steeredLayerCls)
	else:
		class SteeredModel(baseCls):
			def __init__(self, *, vllm_config, prefix: str = prefix):
				super().__init__(vllm_config=vllm_config, prefix=prefix)
				_activate_steering(self, vllm_config, steeredLayerCls)

	SteeredModel.__name__ = f'vllmSteered{baseCls.__name__}'
	SteeredModel.__qualname__ = SteeredModel.__name__
	return SteeredModel


def _activate_steering(model, vllm_config, steeredLayerCls):
	config = vllm_config.model_config.hf_config
	steeredLayers = steered_layers_of(config)
	if not steeredLayers:
		logger.warning('config.steered_layers is empty; %s will not be steered.', type(model).__name__)
	attach_steering(
		model, steeredLayers, hidden_size_of(config),
		steeredLayerCls, dtype=vllm_config.model_config.dtype,
	)
