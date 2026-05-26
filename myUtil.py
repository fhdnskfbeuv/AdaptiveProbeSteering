import csv
import gc
import json
import math
import os.path
import time

import torch
import tqdm
from colorama import Fore, Style
from peft import PeftModel
from strong_reject import evaluate, load_datasets
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoModelForImageTextToText, Qwen3ForCausalLM

import HiddenStateManager
import myJudge

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

nameMap = {
	'thu-coai/Mistral-7B-Instruct-v0.2-safeunlearning': 'Mistral-SU',
	'thu-coai/vicuna-7b-v1.5-safeunlearning': 'Vicuna-SU',
	'lapisrocks/Llama-3-8B-Instruct-TAR-Refusal': 'Llama3-TAR',
	'GraySwanAI/Llama-3-8B-Instruct-RR': 'Llama3-CB',
	'cais/zephyr_7b_r2d2': 'R2D2',
	'Unispac/Llama2-7B-Chat-Augmented': 'Llama2-DA',
	'LLM-LAT/robust-llama3-8b-instruct': 'Llama3-LAT',
	'thkim0305/RepBend_Mistral_7B': 'Mistral-RB',
	'thkim0305/RepBend_Llama3_8B': 'Llama3-RB',
	"GraySwanAI/Mistral-7B-Instruct-RR": 'Mistral-CB',
	'Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment': 'Gemma-DA',
	'Youliang/llama3-8b-instruct-lora-derta-100step': 'Llama3-DeRTA',
	"PawanKrd/Meta-Llama-3-8B-Instruct": 'Llama-3-8B-Instruct',
	'meta-llama/Llama-2-7b-chat-hf': 'Llama-2-7b-chat',
	"Qwen/Qwen3-4B-Instruct-2507": 'Qwen3-4B-Instruct',
	"Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
	"lmsys/vicuna-7b-v1.5": "Vicuna-7b-v1.5",
	"mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
	"google/gemma-2-9b-it": "Gemma-2-9b-it"
}

model2thinkend = {
	'Qwen/Qwen3-4B-Thinking-2507': "</think>",
	'zai-org/GLM-4.6V-Flash': "</think>",
	'Qwen/Qwen3.5-4B': "</think>",
	'Qwen/Qwen3.5-35B-A3B': "</think>",
	'CMU-AIRe/TARS-7B': "</think>",
	'google/gemma-4-E2B-it': "<channel|>",
	'openai/gpt-oss-20b': "<|end|>"
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


def loadModel(modelN, tokenizerN):
	tryNum = 10
	modelN = modelN
	tokenizerN = tokenizerN
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in lora2base.keys():
				print(f'{modelN} has an adapter! Loading now.')
				model = AutoModelForCausalLM.from_pretrained(
					lora2base[modelN], torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa", device_map="auto"
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default")
				config = AutoConfig.from_pretrained(lora2base[modelN], token=os.getenv('HF_TOKEN', default=None))
			else:
				model = AutoModelForCausalLM.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
															 attn_implementation="sdpa" if 'oss' not in modelN else 'eager', device_map="auto")
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None))
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None))
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


def loadVisualModel(modelN, tokenizerN):
	tryNum = 10
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in lora2base.keys():
				print(f'{modelN} has adapter! Loading now.')
				model = AutoModelForImageTextToText.from_pretrained(
					lora2base[modelN], torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa", device_map="auto"
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default")
				config = AutoConfig.from_pretrained(lora2base[modelN], token=os.getenv('HF_TOKEN', default=None))
			else:
				model = AutoModelForImageTextToText.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
																	attn_implementation="sdpa", device_map="auto")
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None))
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None), use_fast_image_processor=True)
			for k in customizedChatTemplate.keys():
				if k in modelN:
					processor.chat_template = customizedChatTemplate[k]
			example = processor.apply_chat_template([{"role": "user", "content": [{'type': 'image'}, {'type': 'text', "text": '{Instruct}'}]}],
													tokenize=True,
													return_tensors="pt",
													return_dict=True,
													add_generation_prompt=True)['input_ids']
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.batch_decode(example, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.convert_ids_to_tokens(example[0])} {Style.RESET_ALL}')
			model.generation_config.use_cache = True
			if model.generation_config.max_length is not None:
				model.generation_config.max_length = None
			if model.generation_config.pad_token_id is None:
				model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] if isinstance(model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id
				print(f"Setting `pad_token_id` to `eos_token_id`:{model.generation_config.pad_token_id} for open-end generation.")
			successDL = True
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
def getLLMEmb(model, hdManager: HiddenStateManager.HDManager, judgeF, biThres, layerIdxs, prompts, processor, maxL, embType, systemPrompt=None):
	completions = []
	avgScore = 0
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {0.0}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			message = [{"role": "user", "content": prompt}]
			if systemPrompt is not None:
				message = [{"role": "system", "content": systemPrompt}] + message
			inputs = processor.apply_chat_template(message,
												   tokenize=True,
												   return_tensors="pt",
												   return_dict=True,
												   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
			# Inference: Generation of the output
			output = model.generate(**inputs, max_new_tokens=maxL,
									output_hidden_states=True, output_logits=True,
									return_dict_in_generate=True, do_sample=False)
			hiddenStates = output.hidden_states  # [maxL, layers + 1, 1, L, D]
			generated_ids = output.sequences
			completion = processor.batch_decode(
				[generated_ids[0][inputs['input_ids'][0].shape[0]:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
			)[0]
			completions.append(completion)
			score = judgeF(prompt, completion)
			avgScore += score
			pbar.set_description(f"{Fore.RED}Average Score: {avgScore / len(completions)}; Current Score: {score}{Style.RESET_ALL}")
			pbar.update()
			if biThres[1] > score > biThres[0]:  # not sure
				continue
			label = int(score >= biThres[1])
			for j in layerIdxs:  # loop for each layer
				if embType == 'last':
					hd = hiddenStates[0][j + 1][:, -1, :].float().clone().cpu()
				elif embType == 'prompt':
					hd = torch.mean(hiddenStates[0][j + 1][0, :, :].float(), dim=0, keepdim=True).clone().cpu()
				elif embType == 'response':
					hd = torch.mean(torch.concat([hiddenStates[t][j + 1][:, -1, :].float() for t in range(len(hiddenStates))], dim=0), dim=0, keepdim=True).clone().cpu()
				elif embType == 'all':
					hd = torch.mean(torch.concat([hiddenStates[t][j + 1][0, :, :].float() for t in range(len(hiddenStates))], dim=0), dim=0, keepdim=True).clone().cpu()
				else:
					print(f'{embType} not implemented')
					exit(1)
				hdManager.add(j, hd, [label] * hd.shape[0])
	return hdManager, completions, avgScore / len(completions)


def easyGen(model, processor, text: str, maxL=128, doSample=False, endThink=None):
	query = [
		{
			"role": "user",
			"content": text
		}
	]
	inputs = processor.apply_chat_template(query,
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
	generated_ids = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample)
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completion = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0]
	fullStr = processor.batch_decode(
		generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
	)[0]
	if endThink is not None:
		completion = completion.split(endThink)[-1]
	return fullStr, completion


def GenAndEval(model, processor, judge, prompts, maxL, doSample=False, endThink=None):
	allCompletion = []
	allScores = []
	meanScore = 0
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			fullStr, completion = easyGen(model, processor, prompt, maxL, doSample, endThink)
			score = judge(prompt, completion)
			allCompletion.append(completion)
			allScores.append(score)
			meanScore = torch.tensor(allScores).float().mean().item()
			pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {allScores[-1]};){Style.RESET_ALL}")
			pbar.update()
	return allCompletion, allScores


def gen(model, processor, prompts, maxL, doSample=False, endThink=None):
	allCompletion = []
	with tqdm.tqdm(prompts, total=len(prompts), dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			fullStr, completion = easyGen(model, processor, prompt, maxL, doSample, endThink)
			allCompletion.append(completion)
			pbar.update()
	return allCompletion


def eval(prPair, judges):
	allScores = {}
	for judgeN in judges:
		judgeM, judgeF = loadJudge(judgeN)
		print(judgeN)
		meanScore = 0
		scores = []
		with tqdm.tqdm(prPair, total=len(prPair), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
			for prompt, response in prPair:
				score = judgeF(prompt, response)
				scores.append(score)
				meanScore = torch.tensor(scores).float().mean().item()
				pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {scores[-1]};){Style.RESET_ALL}")
				pbar.update()
		allScores[judgeN.split(' ')[0]] = scores
		if judgeM is not None:
			del judgeM
			torch.cuda.empty_cache()
			gc.collect()
	return allScores


def loadData(dataName):
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
		insts = loadDataset(r'./instructions/', 50, 50, 50, 50)
		prompts = (insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1])
	else:
		print(f'{dataName} not supported')
		exit(1)
	return prompts


def filterData(model, processor, judge, prompts, maxL, thres, endThink):
	print('Filtering')
	judgeM, judgeF = loadJudge(judge)
	_, scores = GenAndEval(model, processor, judgeF, prompts, maxL, doSample=False, endThink=endThink)
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
