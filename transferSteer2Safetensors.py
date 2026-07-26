"""Merge probe steering vectors (probes.pt) into a safetensors checkpoint.

Supported base architectures: Llama, Mistral, Gemma2, Gemma4 Unified and
Qwen3.5 MoE. The probe steering is expressed as a two-layer MLP (hidden size 1,
ReLU) and attached to the decoder layers, then saved with the model so vLLM can
load it as a normal checkpoint via the matching ``vllmSteered*`` architecture
registered by the ``vllm_steered_plugins`` package (no runtime probes.pt).

Example:
    python transferSteer2Safetensors.py \
        --model meta-llama/Llama-3-8B-Instruct \
        --clfP /path/to/probes.pt \
        --output ./steered
"""

import argparse
import json
import os
import re
import shutil

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

# AutoModelForCausalLM silently loads only the text part of a multimodal model, so
# multimodal ``*ForConditionalGeneration`` models must be loaded through a
# conditional-generation Auto head (which varies by transformers version).
_AUTO_CAUSAL = getattr(transformers, 'AutoModelForCausalLM', None)
_AUTO_MULTIMODAL = [getattr(transformers, name) for name in (
	'AutoModelForConditionalGeneration',
	'AutoModelForImageTextToText',
	'AutoModelForVision2Seq',
) if hasattr(transformers, name)]

# Works both when the plugin package is installed (import root is the inner
# package) and when run straight from the source tree (double-nested path).
# SteeringMLP and the probe-loading helpers all live in _common.
try:
	from vllm_steered_plugins._common import SteeringMLP, getProbe, getTargetScores
except ImportError:
	from vllm_steered_plugins.vllm_steered_plugins._common import SteeringMLP, getProbe, getTargetScores


# HF architecture (config.architectures[0]) -> registered steered vLLM architecture.
ARCH_MAP = {
	'LlamaForCausalLM': 'vllmSteeredLlamaForCausalLM',
	'MistralForCausalLM': 'vllmSteeredMistralForCausalLM',
	'Gemma2ForCausalLM': 'vllmSteeredGemma2ForCausalLM',
	'Gemma4UnifiedForConditionalGeneration': 'vllmSteeredGemma4UnifiedForConditionalGeneration',
	'Qwen3_5MoeForConditionalGeneration': 'vllmSteeredQwen3_5MoeForConditionalGeneration',
	'Qwen3_5ForConditionalGeneration': 'vllmSteeredQwen3_5ForConditionalGeneration',
}

# Multimodal wrappers: the HF->vLLM steering-key mapping is inferred, not tested
# here, so the produced checkpoint must be verified by loading it in vLLM.
MULTIMODAL_ARCHS = {
	'Gemma4UnifiedForConditionalGeneration',
	'Qwen3_5MoeForConditionalGeneration',
	'Qwen3_5ForConditionalGeneration',
}

# Self-contained HF modeling file to copy into the transferred repo, the steered
# class name it defines, and the Auto class for ``auto_map`` (trust_remote_code).
HF_MODEL_FILE = {
	'LlamaForCausalLM': 'SteeredLlamaForCausalLM.py',
	'MistralForCausalLM': 'SteeredMistralForCausalLM.py',
	'Gemma2ForCausalLM': 'SteeredGemma2ForCausalLM.py',
	'Gemma4UnifiedForConditionalGeneration': 'SteeredGemma4UnifiedForConditionalGeneration.py',
	'Qwen3_5MoeForConditionalGeneration': 'SteeredQwen3_5ForConditionalGeneration.py',
	'Qwen3_5ForConditionalGeneration': 'SteeredQwen3_5ForConditionalGeneration.py',
}
HF_CLASS_NAME = {
	'LlamaForCausalLM': 'SteeredLlamaForCausalLM',
	'MistralForCausalLM': 'SteeredMistralForCausalLM',
	'Gemma2ForCausalLM': 'SteeredGemma2ForCausalLM',
	'Gemma4UnifiedForConditionalGeneration': 'SteeredGemma4UnifiedForConditionalGeneration',
	'Qwen3_5MoeForConditionalGeneration': 'SteeredQwen3_5MoeForConditionalGeneration',
	'Qwen3_5ForConditionalGeneration': 'SteeredQwen3_5ForConditionalGeneration',
}
HF_AUTO_CLASS = {
	'LlamaForCausalLM': 'AutoModelForCausalLM',
	'MistralForCausalLM': 'AutoModelForCausalLM',
	'Gemma2ForCausalLM': 'AutoModelForCausalLM',
	'Gemma4UnifiedForConditionalGeneration': 'AutoModelForConditionalGeneration',
	'Qwen3_5MoeForConditionalGeneration': 'AutoModelForConditionalGeneration',
	'Qwen3_5ForConditionalGeneration': 'AutoModelForConditionalGeneration',
}


def sanitizeFilename(name: str) -> str:
	"""Replace characters that are invalid in file/directory names with '_'.

	Keeps alphanumerics, '_', '.' and '-'; strips spaces and characters such as
	``: < > " / \\ | ? *`` that are illegal (notably on Windows).
	"""
	return re.sub(r'[^\w.\-]', '_', name)


def loadModel(modelPath: str, multimodal: bool):
	"""Load the full model with the Auto class matching its architecture.

	``AutoModelForCausalLM`` would silently load only the text part of a multimodal
	model (and drop ``architectures`` from the loaded config), so multimodal models
	are loaded through the conditional-generation Auto classes instead. The
	architecture is read from the on-disk config (``AutoConfig``) by the caller, not
	from the loaded model.
	"""
	candidates = _AUTO_MULTIMODAL if multimodal else [_AUTO_CAUSAL]
	lastErr = None
	for cls in candidates:
		if cls is None:
			continue
		try:
			return cls.from_pretrained(
				modelPath,
				torch_dtype='auto',
				low_cpu_mem_usage=True,
				device_map='cpu',
			)
		except (ValueError, KeyError) as e:
			lastErr = e
			continue
	raise RuntimeError(f'Could not load {modelPath!r}. Last error: {lastErr}')


def resolveLayers(model):
	"""Return the HF ModuleList of decoder layers.

	Tries the common attribute paths for text and multimodal models and returns
	the first that looks like a decoder-layer list (its elements expose ``mlp``).
	"""
	candidates = [
		('model', 'layers'),
		('model', 'language_model', 'layers'),
		('model', 'language_model', 'model', 'layers'),
		('language_model', 'model', 'layers'),
		('language_model', 'layers'),
	]
	for path in candidates:
		obj = model
		for attr in path:
			obj = getattr(obj, attr, None)
			if obj is None:
				break
		if obj is not None and len(obj) > 0 and hasattr(obj[0], 'mlp'):
			return obj
	raise RuntimeError('Could not locate the decoder-layer ModuleList on the HF model.')


def makeReadable(root: str):
	"""Make a saved checkpoint tree readable (dirs 755, files 644).

	``save_pretrained`` honors the process umask, which can leave files at 600;
	this normalizes them so other users/services (e.g. vLLM) can load the model.
	"""
	for dirpath, _dirnames, filenames in os.walk(root):
		os.chmod(dirpath, 0o755)
		for name in filenames:
			os.chmod(os.path.join(dirpath, name), 0o644)


def setArchitectures(saveDir: str, arch: str, autoMap: dict | None = None):
	"""Force the saved config.json to advertise the steered vLLM architecture.

	Some transformers model classes reset ``architectures`` during
	``save_pretrained``, so rewrite the field on disk to be sure vLLM picks the
	registered ``vllmSteered*`` class.  Optionally sets ``auto_map`` so the
	checkpoint can be loaded with ``trust_remote_code=True``.
	"""
	configPath = os.path.join(saveDir, 'config.json')
	with open(configPath, 'r', encoding='utf-8') as f:
		cfg = json.load(f)
	cfg['architectures'] = [arch]
	if autoMap is not None:
		cfg['auto_map'] = autoMap
	with open(configPath, 'w', encoding='utf-8') as f:
		json.dump(cfg, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help='Base HF model (repo id or local path).')
	parser.add_argument('--output', type=str, required=True, help='Output directory for the merged checkpoint.')
	parser.add_argument('--clfP', type=str, required=True, help='Path to the trained probes file (probes.pt).')
	parser.add_argument('--strength', type=str, default='1', help="Target-score strength (passed to getTargetScores).")
	parser.add_argument('--which', type=str, default='best', help="Which probe to use: all/first/best/last.")
	parser.add_argument('--maxShardSize', type=str, default='2GB', help="Max size per safetensors shard (e.g. '500MB', '2GB').")
	args = parser.parse_args()


	print(f'Loading probes from {args.clfP}')
	allProbes = torch.load(args.clfP, map_location='cpu', weights_only=False)
	pm = getProbe(allProbes, args.which)[0]
	strengths = getTargetScores(pm, args.strength)

	# Read the architecture from the on-disk config: AutoModelForCausalLM would load
	# only the text part of a multimodal model and drop ``architectures``.
	config = AutoConfig.from_pretrained(args.model)
	origArch = (getattr(config, 'architectures', None) or [None])[0]
	if origArch not in ARCH_MAP:
		raise RuntimeError(f'Unsupported architecture {origArch!r}. Supported: {sorted(ARCH_MAP)}.')

	print(f'Loading base model {args.model}')
	model = loadModel(args.model, origArch in MULTIMODAL_ARCHS)

	layers = resolveLayers(model)
	numLayers = len(layers)

	steeredLayers = []
	for i in pm.keys():
		# Probe i is trained on the output of layer i, i.e. the stream entering
		# layer i + 1 (matches the vLLM pre-hook / SteeredLlamaDecoderLayer).
		target = i + 1
		if target >= numLayers:
			print(f'Skip probe on layer {i}: target layer {target} out of range ({numLayers} layers).')
			continue
		w = pm[i]['w'].float()
		b = pm[i]['b']
		if isinstance(b, float):
			b = torch.tensor(b)
		whateverPara = next(layers[target].mlp.named_parameters())[1]
		layers[target].steer_mlp = SteeringMLP.from_probe(
			w=w,
			b=b.reshape(-1),
			s=strengths[i].reshape(-1),
			dtype=whateverPara.dtype,
		)
		steeredLayers.append(target)
		print(f'Attached steering MLP to layer {target} (probe on layer {i}).')

	if not steeredLayers:
		raise RuntimeError('No steering vectors were attached; nothing to save.')

	# Tell the vLLM model where the steering MLPs live and which class to use.
	model.config.steered_layers = sorted(steeredLayers)
	model.config.architectures = [ARCH_MAP[origArch]]
	if origArch in MULTIMODAL_ARCHS:
		print(f'WARNING: {origArch} is multimodal; verify the merged checkpoint loads in vLLM '
			  f'(the HF->vLLM steering-key mapping for these is inferred, not tested here).')
	clfName = sanitizeFilename(os.path.split(args.clfP)[-1])
	strengthTag = sanitizeFilename(args.strength)
	saveDir = os.path.join(args.output, args.model + f'_{clfName}_Which{args.which}_Strength{strengthTag}')
	print(f'Saving merged checkpoint to {saveDir}')
	model.save_pretrained(saveDir, safe_serialization=True, max_shard_size=args.maxShardSize)
	# Multimodal models also need the processor/preprocessor configs; AutoProcessor
	# saves those plus the tokenizer. Fall back to the tokenizer for text-only models.
	try:
		from transformers import AutoProcessor
		AutoProcessor.from_pretrained(args.model).save_pretrained(saveDir)
	except Exception:
		AutoTokenizer.from_pretrained(args.model).save_pretrained(saveDir)
	# Copy the self-contained HF modeling file so the checkpoint can be loaded
	# with trust_remote_code=True (e.g. AutoModelForCausalLM.from_pretrained).
	hfFile = HF_MODEL_FILE[origArch]
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	shutil.copy2(os.path.join(scriptDir, 'HFCustomizedModel', hfFile),
				 os.path.join(saveDir, hfFile))
	autoMap = {HF_AUTO_CLASS[origArch]: f'{hfFile[:-3]}.{HF_CLASS_NAME[origArch]}'}
	setArchitectures(saveDir, ARCH_MAP[origArch], autoMap)
	makeReadable(saveDir)
	print(f'Done. Steered layers: {model.config.steered_layers}')
	print(f'Point vLLM at this directory (not --output itself):\n    {os.path.abspath(saveDir)}')


if __name__ == '__main__':
	main()
