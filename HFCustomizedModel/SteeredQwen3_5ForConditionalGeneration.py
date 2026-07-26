"""Steered Qwen3.5 (dense and MoE, multimodal) for Transformers (HF).

Self-contained: loads a merged checkpoint produced by ``transferSteer2Safetensors.py``
and applies the baked probe steering (a 2-layer MLP, hidden size 1, ReLU) on the
layers listed in ``config.steered_layers``. Usage::

    from SteeredQwen3_5ForConditionalGeneration import SteeredQwen3_5ForConditionalGeneration
    model = SteeredQwen3_5ForConditionalGeneration.from_pretrained(steered_dir)
"""

import logging

import torch
from torch import nn
from transformers import (
	Qwen3_5ForConditionalGeneration,
	Qwen3_5MoeForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class SteeringMLP(nn.Module):
	def __init__(self, hiddenSize: int, dtype: torch.dtype | None = None):
		super().__init__()
		self.down = nn.Linear(hiddenSize, 1, bias=True, dtype=dtype)
		self.up = nn.Linear(1, hiddenSize, bias=False, dtype=dtype)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.up(torch.relu(self.down(x)))


def _resolve_layers(model):
	for path in [('model', 'layers'), ('model', 'language_model', 'layers'),
				 ('model', 'language_model', 'model', 'layers'),
				 ('language_model', 'model', 'layers'), ('language_model', 'layers')]:
		obj = model
		for attr in path:
			obj = getattr(obj, attr, None)
			if obj is None:
				break
		if obj is not None and len(obj) > 0 and hasattr(obj[0], 'mlp'):
			return obj
	raise RuntimeError('Could not locate the decoder-layer ModuleList.')


def _steer_pre_hook(module, args, kwargs):
	mlp = module.steer_mlp
	if 'hidden_states' in kwargs:
		hidden = kwargs['hidden_states']
		kwargs = dict(kwargs)
		kwargs['hidden_states'] = hidden + mlp(hidden)
	else:
		hidden = args[0]
		args = (hidden + mlp(hidden),) + tuple(args[1:])
	return args, kwargs


def _attach_steering(model):
	config = model.config
	steeredLayers = list(getattr(config, 'steered_layers', None) or [])
	if not steeredLayers:
		logger.warning('config.steered_layers is empty; no steering applied.')
		return
	layers = _resolve_layers(model)
	hiddenSize = getattr(config, 'hidden_size', None)
	if hiddenSize is None:
		textConfig = getattr(config, 'text_config', None)
		if textConfig is not None:
			hiddenSize = textConfig.hidden_size
	for target in steeredLayers:
		if target >= len(layers):
			continue
		layer = layers[target]
		if getattr(layer, 'steer_mlp', None) is not None:
			continue
		firstPara = next(layer.parameters(), None)
		layerDtype = firstPara.dtype if firstPara is not None else None
		layer.steer_mlp = SteeringMLP(hiddenSize, dtype=layerDtype)
		layer.register_forward_pre_hook(_steer_pre_hook, with_kwargs=True)


class SteeredQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		_attach_steering(self)


class SteeredQwen3_5MoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		_attach_steering(self)
