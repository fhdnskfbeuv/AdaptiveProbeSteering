import copy
import csv
import gc
import json
import math
import os.path
import time

import numpy as np
import torch
import tqdm
from colorama import Fore, Style
from peft import PeftModel
from strong_reject import evaluate, load_datasets
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoModelForImageTextToText, Gemma4UnifiedProcessor

import HiddenStateManager
import myJudge

USE_PROCESSOR = "use_processor"

customizedChatTemplate = {  # We hope authors of models below provide official jinja-format chat template :(
	'Youliang/llama3-8b-instruct-lora-derta-100step': """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% else %}{% endif %}{% endfor %}""",
	# Provided in their hf repo. Yet, we must say that this is, in fact, similar to Llama2 and is quite different from Llama3's fromat:( Oh my god! They use Llama3's bos token but use Llama2's format????
	"vicuna-7b-v1.5": """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}""",
	#  Provided in their hf repo.
	# "vicuna-7b-v1.5": """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] | trim + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] | trim + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
}

lora2base = {
	'Youliang/llama3-8b-instruct-lora-derta-100step': "PawanKrd/Meta-Llama-3-8B-Instruct",
	"thkim0305/RepBend_Llama3_8B_LoRA": "PawanKrd/Meta-Llama-3-8B-Instruct",
	"thkim0305/RepBend_Mistral_7B_LoRA": "mistralai/Mistral-7B-Instruct-v0.2"
}

model2thinkend = {
	'Qwen/Qwen3-4B-Thinking-2507': "</think>",
	'zai-org/GLM-4.6V-Flash': "</think>",
	'Qwen/Qwen3.5-4B': "</think>",
	'Qwen/Qwen3.5-35B-A3B': "</think>",
	'Qwen/Qwen3.6-35B-A3B': "</think>",
	'CMU-AIRe/TARS-7B': "</think>",
	'wangzhang/Qwen3.5-35B-A3B-abliterated': "</think>",
	'Qwen/Qwen3.6-27B': "</think>",
	'google/gemma-4-12B-it': USE_PROCESSOR,
	'google/gemma-4-E2B-it': USE_PROCESSOR
}


def extractFromJson(file_path, key):
	values = []
	with open(file_path, 'r', encoding='utf-8') as file:
		data = json.load(file)
		if not isinstance(data, list):
			data = [data]
		for d in data:
			values.append(d[key])
	return values


def loadDataset(dirP, harmTrain, benignTrain, harmVal, benignVal, full=False, trainSize=None):
	harmfulTrain = extractFromJson(os.path.join(dirP, 'harmful_train.json'), "instruction")
	harmfulVal = extractFromJson(os.path.join(dirP, 'harmful_val.json'), "instruction")
	harmfulTest = extractFromJson(os.path.join(dirP, 'harmful_test.json'), "instruction")
	harmlessTrain = extractFromJson(os.path.join(dirP, 'harmless_train.json'), "instruction")
	harmlessVal = extractFromJson(os.path.join(dirP, 'harmless_val.json'), "instruction")
	harmlessTest = extractFromJson(os.path.join(dirP, 'harmless_test.json'), "instruction")
	harmfulTVLen = len(harmfulTrain + harmfulVal)
	harmlessTVLen = len(harmlessTrain + harmlessVal)
	seenHarmful = harmfulTrain + harmfulVal
	seenHarmless = harmlessTrain + harmlessVal
	if full:
		if trainSize is not None:
			totalNum = len(seenHarmful + harmfulTest)
			tNum = int(totalNum * trainSize)
			return {
				'train': [(seenHarmful + harmfulTest)[:tNum], (seenHarmless + harmlessTest)[:tNum]],
				'val': [(seenHarmful + harmfulTest)[tNum:], (seenHarmless + harmlessTest)[tNum:totalNum]],
				'test': [harmfulTest, harmlessTest],
			}
		else:
			return {
				'train': [seenHarmful + harmfulTest, (seenHarmless + harmlessTest)[:len(seenHarmful + harmfulTest)]],
				'val': [seenHarmful + harmfulTest, (seenHarmless + harmlessTest)[:len(seenHarmful + harmfulTest)]],
				'test': [harmfulTest, harmlessTest],
			}
	return {
		'train': [seenHarmful[:harmTrain], seenHarmless[:benignTrain]],
		'val': [seenHarmful[harmTrain:min(harmfulTVLen, harmTrain + harmVal)], seenHarmless[benignTrain:min(harmlessTVLen, benignTrain + benignVal)]],
		'test': [harmfulTest, harmlessTest],
	}


def loadModel(modelN, tokenizerN, trust_remote_code=False):
	tryNum = 10
	modelN = modelN
	tokenizerN = tokenizerN
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in lora2base.keys():
				print(f'{modelN} has an adapter! Loading now.')
				model = AutoModelForCausalLM.from_pretrained(
					lora2base[modelN], torch_dtype='auto', token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa", device_map="auto",
					trust_remote_code=trust_remote_code
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default",
												  trust_remote_code=trust_remote_code)
				config = AutoConfig.from_pretrained(lora2base[modelN], token=os.getenv('HF_TOKEN', default=None),
													trust_remote_code=trust_remote_code)
			else:
				model = AutoModelForCausalLM.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
															 attn_implementation="sdpa" if 'oss' not in modelN else 'eager', device_map="auto",
															 trust_remote_code=trust_remote_code)
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None),
													trust_remote_code=trust_remote_code)
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None),
													  trust_remote_code=trust_remote_code)
			processor.padding_side = 'left'
			if 'r2d2' in modelN.lower():
				processor.chat_template = '{{ bos_token }}' + processor.chat_template
			for k in customizedChatTemplate.keys():
				if k in modelN:
					processor.chat_template = customizedChatTemplate[k]
			example = processor.apply_chat_template([{"role": "user", "content": '{Instruct}'}],
													tokenize=True,
													return_tensors="pt",
													return_dict=True,
													add_generation_prompt=True)['input_ids']
			print(f'{Fore.RED} {modelN}: {type(model)} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.batch_decode(example, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.convert_ids_to_tokens(example[0])} {Style.RESET_ALL}')
			model.generation_config.use_cache = True
			if model.generation_config.max_length is not None:
				model.generation_config.max_length = None
			if model.generation_config.pad_token_id is None:
				model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] if isinstance(model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id
				print(f"Setting `pad_token_id` to `eos_token_id`:{model.generation_config.pad_token_id} for open-end generation.")
			time.sleep(1)
			if processor.pad_token_id is None:
				processor.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
			tryNum = 0
		except Exception as e:
			print(e)
			print('What can I say?')
			time.sleep(1)
	return model, processor, config


def loadVisualModel(modelN, tokenizerN, trust_remote_code=False):
	tryNum = 10
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in lora2base.keys():
				print(f'{modelN} has adapter! Loading now.')
				model = AutoModelForImageTextToText.from_pretrained(
					lora2base[modelN], torch_dtype='auto', token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa", device_map="auto",
					trust_remote_code=trust_remote_code
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default",
												  trust_remote_code=trust_remote_code)
				config = AutoConfig.from_pretrained(lora2base[modelN], token=os.getenv('HF_TOKEN', default=None),
													trust_remote_code=trust_remote_code)
			else:
				model = AutoModelForImageTextToText.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
																	attn_implementation="sdpa", device_map="auto",
																	trust_remote_code=trust_remote_code)
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None),
													trust_remote_code=trust_remote_code)
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None), use_fast_image_processor=True,
													  trust_remote_code=trust_remote_code)
			processor.tokenizer.padding_side = 'left'
			for k in customizedChatTemplate.keys():
				if k in modelN:
					processor.chat_template = customizedChatTemplate[k]
			example = processor.apply_chat_template([{"role": "user", "content": [{'type': 'text', "text": '{Instruct}'}]}],
													tokenize=True,
													return_tensors="pt",
													return_dict=True,
													add_generation_prompt=True)['input_ids']
			print(f'{Fore.RED} {modelN}: {type(model)} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.batch_decode(example, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.convert_ids_to_tokens(example[0])} {Style.RESET_ALL}')
			model.generation_config.use_cache = True
			if model.generation_config.max_length is not None:
				model.generation_config.max_length = None
			if model.generation_config.pad_token_id is None:
				model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] if isinstance(model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id
				print(f"Setting `pad_token_id` to `eos_token_id`:{model.generation_config.pad_token_id} for open-end generation.")
			time.sleep(1)
			if processor.tokenizer.pad_token_id is None:
				processor.tokenizer.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
			tryNum = 0
		except Exception as e:
			print(e)
			print('What can I say?')
			time.sleep(1)
	return model, processor, config


def loadJudge(judge):
	tryNum = 10
	judgeM = None
	judgeF = None
	judgeN = judge.split(' ')
	while tryNum > 0:
		try:
			tryNum -= 1
			if judgeN[0] == 'hb':
				judgeF = myJudge.HarmBenchJudge
				myJudge.HarmBenchJudge('a', 'b')
				judgeM = evaluate.cached_models["harmbench"][0]
			elif judgeN[0] == 'srf':
				judgeF = myJudge.StrongRejectJudge
				myJudge.StrongRejectJudge('a', 'b')
				judgeM = evaluate.cached_models["strongreject_finetuned"][0]
			elif len(judgeN) == 3:
				judgeF = myJudge.SJRubricAPI(judgeN[0], judgeN[1], judgeN[2]).judge
			else:
				print(f'{judgeN} not implemented')
				exit(1)
			tryNum = 0
		except Exception as e:
			print(e)
	return judgeM, judgeF


@torch.no_grad()
def getStartAndEnd(attentionMask, allIDs, inputLen, eosID):  # (B, L) aligned with input, end points to the last input
	startIdxs = attentionMask.argmax(dim=-1)
	eos_positions = (allIDs[:, inputLen:] == eosID).int().argmax(dim=-1) - 1 + inputLen
	has_eos = (allIDs[:, inputLen:] == eosID).any(dim=-1)
	endIdxs = torch.where(
		has_eos,
		eos_positions,
		allIDs.shape[1] - 2
	)
	return startIdxs, endIdxs  # (B,)


@torch.no_grad()
def getLLMEmb(model, hdManager: HiddenStateManager.HDManager, judgeF, biThres, layerIdxs, prompts, processor, maxL, batchSize, embType, hooks, systemPrompt=None):
	completions = []
	avgScore = 0
	allScores = []
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {0.0}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompts = prompts[left:min(len(prompts), left + batchSize)]
			messages = [[{"role": "user", "content": prompt}] for prompt in batchedPrompts]
			if systemPrompt is not None:
				messages = [[{"role": "system", "content": systemPrompt}] + message for message in messages]
			inputs = processor.apply_chat_template(messages,
												   tokenize=True,
												   padding=True,
												   truncation=True,
												   return_tensors="pt",
												   return_dict=True,
												   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
			# Inference: Generation of the output
			output = model.generate(**inputs, max_new_tokens=maxL,
									output_hidden_states=True, output_logits=('top' in embType),
									return_dict_in_generate=True, do_sample=False)
			hiddenStates = output.hidden_states  # [maxL, layers + 1, B, L, D]
			generated_ids = output.sequences  # (B, L + 1 because shift)
			startIdxs, endIdxs = getStartAndEnd(inputs['attention_mask'], generated_ids, inputs['input_ids'].shape[1], processor.eos_token_id)
			batchCompletions = processor.batch_decode(generated_ids[:, inputs['input_ids'][0].shape[0]:],
													  skip_special_tokens=True, clean_up_tokenization_spaces=False)
			completions += batchCompletions
			batchScores = judgeF(batchedPrompts, batchCompletions)
			allScores += batchScores
			avgScore = np.array(allScores).mean().item()
			for i in range(len(batchedPrompts)):
				pbar.set_description(f"{Fore.RED}Average Score: {avgScore}; Current Score: {batchScores[i]}{Style.RESET_ALL}")
				pbar.update()
				pbar.refresh()
			labels = [-1 if (biThres[1] > score > biThres[0]) else int(score >= biThres[1]) for score in batchScores]
			if 'top' in embType and hooks is not None:
				hooks[-1].enable = False
				importantIndices = tokenImportanceKL(model, processor, output, copy.deepcopy(inputs), embType.split('_')[0], float(embType.split('_')[1]), labels)
				hooks[-1].enable = True
			for i in range(generated_ids.shape[0]):
				if labels[i] == -1:
					continue
				# print(f'Select {len(importantIndices[0])} tokens: {importantIndices}')
				for j in layerIdxs:  # loop for each layer
					if embType == 'last':
						hd = hiddenStates[0][j + 1][i:i + 1, -1, :].float().clone().cpu()
					elif 'top' in embType and hooks is not None:
						hd = torch.mean(torch.concat([hiddenStates[t][j + 1][i:i + 1, -1, :] for t in range(len(hiddenStates))], dim=0).float()[importantIndices[i], :], dim=0, keepdim=True).cpu()
					elif 'top' in embType and hooks is None:
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[inputs['input_ids'][i].shape[0] - 1:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					elif embType == 'prompt':
						hd = torch.mean(hiddenStates[0][j + 1][i, startIdxs[i]:, :].float(), dim=0, keepdim=True).clone().cpu()
					elif embType == 'response':
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[inputs['input_ids'][i].shape[0] - 1:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					elif embType == 'all':
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[startIdxs[i]:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					else:
						print(f'{embType} not implemented')
						exit(1)
					hdManager.add(j, hd, [labels[i]] * hd.shape[0])
	return hdManager, completions, avgScore


@torch.no_grad()
def getLVLMEmb(model, hdManager: HiddenStateManager.HDManager, judgeF, biThres, layerIdxs, prompts, processor, maxL, batchSize, embType, hooks, systemPrompt=None):
	completions = []
	avgScore = 0
	allScores = []
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {0.0}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompts = prompts[left:min(len(prompts), left + batchSize)]
			messages = [[
				{
					"role": "user",
					"content": [{'type': 'text', "text": prompt}]
				}
			] for prompt in batchedPrompts]
			if systemPrompt is not None:
				messages = [[{"role": "system", "content": [{'type': 'text', "text": systemPrompt}]}] + message for message in messages]
			inputs = processor.apply_chat_template(messages,
												   tokenize=True,
												   padding=True,
												   truncation=True,
												   return_tensors="pt",
												   return_dict=True,
												   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
			# Inference: Generation of the output
			output = model.generate(**inputs, max_new_tokens=maxL,
									output_hidden_states=True, output_logits=True,
									return_dict_in_generate=True, do_sample=False)
			hiddenStates = output.hidden_states  # [maxL, layers + 1, 1, L, D]
			generated_ids = output.sequences
			startIdxs, endIdxs = getStartAndEnd(inputs['attention_mask'], generated_ids, inputs['input_ids'].shape[1], processor.tokenizer.eos_token_id)
			batchCompletions = processor.batch_decode(generated_ids[:, inputs['input_ids'][0].shape[0]:],
													  skip_special_tokens=True, clean_up_tokenization_spaces=False)
			completions += batchCompletions
			batchScores = judgeF(batchedPrompts, batchCompletions)
			allScores += batchScores
			avgScore = np.array(allScores).mean().item()
			for i in range(len(batchedPrompts)):
				pbar.set_description(f"{Fore.RED}Average Score: {avgScore}; Current Score: {batchScores[i]}{Style.RESET_ALL}")
				pbar.update()
				pbar.refresh()
			labels = [-1 if (biThres[1] > score > biThres[0]) else int(score >= biThres[1]) for score in batchScores]
			if 'top' in embType and hooks is not None:
				hooks[-1].enable = False
				importantIndices = tokenImportanceKL(model, processor, output, copy.deepcopy(inputs), embType.split('_')[0], float(embType.split('_')[1]), labels)
				hooks[-1].enable = True
			for i in range(generated_ids.shape[0]):
				if labels[i] == -1:
					continue
				# print(f'Select {len(importantIndices[0])} tokens: {importantIndices}')
				for j in layerIdxs:  # loop for each layer
					if embType == 'last':
						hd = hiddenStates[0][j + 1][i:i + 1, -1, :].float().clone().cpu()
					elif 'top' in embType and hooks is not None:
						hd = torch.mean(torch.concat([hiddenStates[t][j + 1][i:i + 1, -1, :] for t in range(len(hiddenStates))], dim=0).float()[importantIndices[i], :], dim=0, keepdim=True).cpu()
					elif 'top' in embType and hooks is None:
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[inputs['input_ids'][i].shape[0] - 1:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					elif embType == 'prompt':
						hd = torch.mean(hiddenStates[0][j + 1][i, startIdxs[i]:, :].float(), dim=0, keepdim=True).clone().cpu()
					elif embType == 'response':
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[inputs['input_ids'][i].shape[0] - 1:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					elif embType == 'all':
						hd = torch.concat([hiddenStates[t][j + 1][i, :, :] for t in range(len(hiddenStates))], dim=0).float()[startIdxs[i]:endIdxs[i] + 1, :]
						hd = torch.mean(hd, dim=0, keepdim=True).cpu()
					else:
						print(f'{embType} not implemented')
						exit(1)
					hdManager.add(j, hd, [labels[i]] * hd.shape[0])
	return hdManager, completions, avgScore


@torch.no_grad()
def easyGen(model, processor, prompts: list[str], maxL=128, doSample=False, endThink=None):
	queries = [[{"role": "user", "content": prompt}] for prompt in prompts]
	inputs = processor.apply_chat_template(queries,
										   padding=True,
										   truncation=True,
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
	generated_ids = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample)
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completions = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True if (endThink is None or endThink != USE_PROCESSOR) else False, clean_up_tokenization_spaces=False
	)
	fullStrs = processor.batch_decode(
		generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
	)
	if endThink is not None:
		if endThink == USE_PROCESSOR:
			completions = [processor.parse_response(completion)["response"] for completion in completions]
		else:
			completions = [completion.split(endThink)[-1] for completion in completions]
	return fullStrs, completions


@torch.no_grad()
def easyLVLMGen(model, processor, prompts: list[str], imgs: list, maxL=128, doSample=False, rawCompletion=False, endThink=None):
	queries = [[
		{
			"role": "user",
			"content": [{'type': 'image', "image": img}, {'type': 'text', "text": prompt}] if img is not None else [{'type': 'text', "text": prompt}]
		}
	] for prompt, img in zip(prompts, imgs)]
	inputs = processor.apply_chat_template(queries,
										   truncation=True,
										   padding=True,
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
	output = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample, return_dict_in_generate=True)
	generated_ids = output.sequences
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completions = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True if (endThink is None or endThink != USE_PROCESSOR) else False, clean_up_tokenization_spaces=False
	)
	fullStrs = processor.batch_decode(
		generated_ids, skip_special_tokens=not rawCompletion, clean_up_tokenization_spaces=False
	)
	if endThink is not None:
		if endThink == USE_PROCESSOR:
			completions = [processor.parse_response(completion)["response"] for completion in completions]
		else:
			completions = [completion.split(endThink)[-1] for completion in completions]
	return fullStrs, completions


@torch.no_grad()
def GenAndEval(model, processor, judgeF, prompts, maxL, batchSize, doSample=False, endThink=None):
	allCompletion = []
	allScores = []
	meanScore = 0
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompts = prompts[left:min(len(prompts), left + batchSize)]
			fullStrs, completions = easyGen(model, processor, batchedPrompts, maxL, doSample, endThink)
			allCompletion += completions
			scores = judgeF(batchedPrompts, completions)
			allScores += scores
			meanScore = np.array(allScores).mean().item()
			for i in range(len(batchedPrompts)):
				pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {scores[i]}{Style.RESET_ALL}")
				pbar.update()
				pbar.refresh()
	return allCompletion, allScores


@torch.no_grad()
def GenAndEvalLVLM(model, processor, judgeF, prompts, maxL, batchSize, imgs=None, doSample=False, endThink=None):
	allCompletion = []
	allScores = []
	meanScore = 0
	imgs = imgs if imgs is not None else [None] * len(prompts)
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompts = prompts[left:min(len(prompts), left + batchSize)]
			fullStrs, completions = easyLVLMGen(model, processor,
												batchedPrompts,
												imgs[left:min(len(prompts), left + batchSize)],
												maxL, doSample, False, endThink)
			allCompletion += completions
			scores = judgeF(batchedPrompts, completions)
			allScores += scores
			meanScore = np.array(allScores).mean().item()
			for i in range(len(batchedPrompts)):
				pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {scores[i]}{Style.RESET_ALL}")
				pbar.update()
				pbar.refresh()
	return allCompletion, allScores


@torch.no_grad()
def gen(model, processor, prompts, maxL, batchSize, doSample=False, endThink=None):
	allCompletion = []
	with tqdm.tqdm(prompts, total=len(prompts), dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompt = prompts[left:min(len(prompts), left + batchSize)]
			fullStrs, completions = easyGen(model, processor, batchedPrompt, maxL, doSample, endThink)
			for completion in completions:
				allCompletion.append(completion)
				pbar.update()
				pbar.refresh()
	return allCompletion


@torch.no_grad()
def genLVLM(model, processor, prompts, imgs, maxL, batchSize, doSample=False, endThink=None):
	allCompletion = []
	imgs = imgs if imgs is not None else [None] * len(prompts)
	with tqdm.tqdm(prompts, total=len(prompts), dynamic_ncols=True) as pbar:
		for left in range(0, len(prompts), batchSize):
			batchedPrompt = prompts[left:min(len(prompts), left + batchSize)]
			batchedImg = imgs[left:min(len(prompts), left + batchSize)]
			fullStrs, completions = easyLVLMGen(model, processor,
												batchedPrompt,
												batchedImg,
												maxL, doSample, False, endThink)
			for completion in completions:
				allCompletion.append(completion)
				pbar.update()
				pbar.refresh()
	return allCompletion


@torch.no_grad()
def eval(prompts, responses, judges, batchSize):
	allScores = {}
	assert len(prompts) == len(responses)
	for judgeN in judges:
		judgeM, judgeF = loadJudge(judgeN)
		print(judgeN)
		success = False
		initialBs = copy.deepcopy(batchSize)
		while success is False:
			try:
				meanScore = 0
				scores = []
				with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
					for left in range(0, len(prompts), initialBs):
						batchedPrompt = prompts[left:min(len(prompts), left + initialBs)]
						batchedResponses = responses[left:min(len(responses), left + initialBs)]
						batchedScores = judgeF(batchedPrompt, batchedResponses)
						scores += batchedScores
						meanScore = np.array(scores).mean().item()
						for i in range(len(batchedScores)):
							pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {batchedScores[i]};){Style.RESET_ALL}")
							pbar.update()
							pbar.refresh()
				success = True
				allScores[judgeN.split(' ')[0]] = scores
			except torch.cuda.OutOfMemoryError as e:
				print(e)
				if initialBs == 1:
					print('Input is too long.')
					success = True
				initialBs = max(1, initialBs // 2)
				gc.collect()
				torch.cuda.empty_cache()
		if judgeM is not None:
			del judgeM
			torch.cuda.empty_cache()
			gc.collect()
	return allScores


def loadData(dataName, full=False):
	prompts = []
	if dataName == 'sr':
		prompts = [p['forbidden_prompt'] for p in load_datasets.load_strongreject()]
		prompts = prompts
	elif dataName == 'harmbench':
		csvr = csv.reader(open(r'./instructions/harmbench_behaviors_text_test.csv', 'r+'))
		for row in csvr:
			if row[1] == 'standard':
				prompts.append(row[0])
		prompts = prompts
	elif dataName == 'harm':
		csvr = csv.reader(open(r'./instructions/harmbench_behaviors_text_test.csv', 'r+'))
		for row in csvr:
			if row[1] == 'standard':
				prompts.append(row[0])
		prompts = prompts[:100] + [p['forbidden_prompt'] for p in load_datasets.load_strongreject()][:100]
	elif dataName == 'benign':
		insts = loadDataset(r'./instructions/', 100, 100, 100, 100)
		prompts = insts['train'][1]
	elif dataName == 'train':
		insts = loadDataset(r'./instructions/', 50, 50, 50, 50, full=full)
		prompts = (insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1])
	else:
		print(f'{dataName} not supported')
		exit(1)
	return prompts


@torch.no_grad()
def filterData(model, processor, judge, prompts, maxL, batchSize, thres, endThink):
	print('Filtering')
	judgeM, judgeF = loadJudge(judge)
	_, scores = GenAndEval(model, processor, judgeF, prompts, maxL, batchSize, doSample=False, endThink=endThink)
	del judgeM
	ret = []
	for i, p in enumerate(prompts):
		if scores[i] >= thres:
			ret.append(p)
	return ret


def getLayer(layerNum, layer: list):
	assert len(layer) <= 2
	if 0 <= layer[-1] <= 1:
		if len(layer) < 2:
			layer.insert(0, 0.0)
		layerIdxs = list(range(layerNum))[math.floor(layerNum * layer[0]):max(math.floor(layerNum * layer[0]) + 1, math.ceil(layerNum * layer[1]))]
	else:
		if len(layer) == 1:
			layer.insert(0, -layerNum)
		layerIdxs = list(range(layerNum))[layerNum + int(layer[0]):layerNum + 1 + int(layer[1])]
	return layerIdxs


@torch.no_grad()
def getInputsAndLabels(processor, prompt, target, img):
	inputNoTarget = processor.apply_chat_template(
		[
			{"role": "user", "content": [{'type': 'image', 'image': img}, {'type': 'text', "text": prompt}]},
		],
		tokenize=True,
		add_generation_prompt=True,
		return_attention_mask=True,
		return_tensors="pt",
		return_dict=True,
	
	)
	inputWithTarget = processor.apply_chat_template(
		[
			{"role": "user", "content": [{'type': 'image', 'image': img}, {'type': 'text', "text": prompt}]},
			{"role": "assistant", "content": [{'type': 'text', "text": target}]}
		],
		tokenize=True,
		add_generation_prompt=False,
		continue_final_message=True,
		return_attention_mask=True,
		return_tensors="pt",
		return_dict=True,
	
	)
	label = inputWithTarget.input_ids.clone()
	label[:, :inputNoTarget.input_ids.shape[-1]] = -100
	return inputWithTarget, label


@torch.no_grad()
def top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
	sorted_probs, sorted_indices = torch.sort(probs, descending=True)
	cumulative_probs = torch.cumsum(sorted_probs, dim=0)
	mask = cumulative_probs <= p
	if not mask.any():
		mask[0] = True  # Keep at least the top token
	selected_sorted_indices = sorted_indices[mask]
	return selected_sorted_indices


@torch.no_grad()
def tokenImportanceKL(model, processor, fullOutput, originalInput, mode, kp, labels):
	with torch.no_grad():
		retIndices = []
		fullOutputIDs = fullOutput.sequences[:, :-1]  # [B, L], the last ID has no GT
		fullOutputLogits = torch.stack([l for l in fullOutput.logits], dim=1)  # [B, maxL, C]
		originalInput['input_ids'] = fullOutputIDs.to(originalInput['input_ids'])
		if hasattr(originalInput, 'attention_mask'):
			newMask = ((originalInput['input_ids'] != processor.eos_token_id) and (originalInput['input_ids'] != processor.pad_token_id)).to(originalInput['attention_mask'])
			newMask[:, :originalInput['attention_mask'].shape[1]] = originalInput['attention_mask'].clone().detach()
			originalInput['attention_mask'] = newMask
		else:
			originalInput['attention_mask'] = ((originalInput['input_ids'] != processor.eos_token_id) and (originalInput['input_ids'] != processor.pad_token_id)).to(torch.long).to(originalInput['input_ids'].device)
		noSteerLogits = model(**originalInput.to(model.device)).logits  # [B, L, C]
		noSteerLogits = noSteerLogits[:, -fullOutputLogits.shape[1]:, :]
		# assert noSteerLogits.shape[1] == fullOutputLogits.shape[1]
		for i in range(noSteerLogits.shape[0]):
			kl = torch.nn.functional.kl_div(torch.log_softmax(fullOutputLogits[i].float().to(noSteerLogits.device), dim=-1), noSteerLogits[i].float().softmax(dim=-1), reduction='none').sum(dim=-1)  # [L]
			if labels[i] == 0:
				kl = torch.max(kl) - kl
			kl[originalInput['attention_mask'][i][-fullOutputLogits.shape[1]:] == 0] = 0.0
			# kl[:startIndices[i] - 1] = 0.0
			if mode == 'topk':
				_, index = torch.topk(kl, min(int(kp), torch.sum(originalInput['attention_mask'][i][-fullOutputLogits.shape[1]:], dtype=torch.long)))
				retIndices.append(index.cpu())
			elif mode == 'topp':
				kl /= kl.sum(dim=-1)
				retIndices.append(top_p(kl, kp).cpu())
			else:
				print(f'{mode} not supported')
				exit(1)
	# retIndices[-1] = retIndices[-1] + startIndices[i] - 1
	return retIndices
