import argparse
import logging
import os

import torch
from datasets import disable_caching

import HiddenStateManager
import ProbeManager
import myUtil

if __name__ == '__main__':
	logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help="The repo name on huggingface")
	parser.add_argument('--tokenizer', type=str, help="Some repos do not provide tokenizer. So you have to give the repo that contain the tokenizer")
	parser.add_argument('--pt', type=str, required=True, help="If you want to use the percentile to determine target logits, type in a float [0, 1] (e.g., \"1.0\"). If you want to specify the target logits, type in \"abs[float]\" (e.g., \"abs0\")")
	parser.add_argument('--evalPT', type=str, help="If you want to use the percentile to determine target logits, type in a float [0, 1] (e.g., \"1.0\"). If you want to specify the target logits, type in \"abs[float]\" (e.g., \"abs0\")")
	parser.add_argument('--thres', type=float, required=True, nargs='+', help="The low and the high threshold for the annotator (e.g., 0.05 0.6)")
	parser.add_argument('--maxIter', type=int, required=True, help="The maximum iteration of adaptive retraining")
	parser.add_argument('--bs', type=int, default=1, help="Batch size")
	parser.add_argument('--trainL', type=int, required=True, help="the max new token during adaptive retraining")
	parser.add_argument('--embType', type=str, required=True)
	parser.add_argument('--saveDir', type=str, required=True, help="The root of probe storage")
	parser.add_argument('--judge', type=str, required=True, help="The annotator")
	parser.add_argument('--layer', nargs='+', type=float, default=[-2], help="The interval of selected layer. Length <= 2")
	parser.add_argument('--linearC', type=str, required=True, choices=['cuLR', 'cuSVC', 'skLR'], help="The type of linear model")
	parser.add_argument('--trust', action='store_true', help="Trust remote code?")
	parser.add_argument('--val', action='store_true', help="Whether to conduct validation during adaptive retraining")
	parser.add_argument('--normReg', action='store_true', help="Whether to dynamically set regularization strength according to norm of inputs")
	parser.add_argument('--filter', action='store_true', help="Whether to use StrongReject's finetuned judge to filter out benign prompts that are refused by the model")
	args = parser.parse_args()
	print(args)
	assert args.thres[1] >= args.thres[0] and args.thres[1] <= 1.0 and args.thres[0] >= 0.0
	if args.tokenizer is None:
		args.tokenizer = args.model
	
	# load model & processor
	disable_caching()
	model, processor, config = myUtil.loadVisualModel(args.model, args.tokenizer, args.trust)
	saveDir = os.path.join(args.saveDir, args.model.replace('./', '').replace('/', '_'))
	os.makedirs(saveDir, exist_ok=True)
	judgeN = args.judge.split(' ')[0]
	layerIdxs = myUtil.getLayer(config.text_config.num_hidden_layers, args.layer)
	clfP = os.path.join(saveDir,
						f'judge{judgeN}_embType{args.embType}_filter{args.filter}_normReg{args.normReg}_layer[{layerIdxs[0]}, {layerIdxs[-1]}]_linearC{args.linearC}_maxIter{args.maxIter}_trainL{args.trainL}_pt{args.pt}_softThres{args.thres}.pt'.replace(
							'/',
							'-'))
	harmTrainPrompts, benignTrainPrompts, harmValPrompts, _ = myUtil.loadData('train')
	if args.filter:
		benignTrainPrompts = myUtil.filterData(model, processor, args.judge, benignTrainPrompts, 512, args.bs, args.thres[1], None)
	# get initial embd
	hdManager = HiddenStateManager.HDManager(layerIdxs)
	# get the hd you don't prefer
	hdManager, _, _ = myUtil.getLVLMEmb(model, hdManager, lambda x, y: [0.0] * len(x), args.thres, layerIdxs,
									   harmTrainPrompts, processor,
									   args.trainL if (args.embType in ['all', 'response'] or 'top' in args.embType) else 1,
									   args.bs, args.embType, None, None)
	# get the hd you prefer
	hdManager, _, _ = myUtil.getLVLMEmb(model, hdManager, lambda x, y: [1.0] * len(x), args.thres, layerIdxs,
									   benignTrainPrompts, processor,
									   args.trainL if (args.embType in ['all', 'response'] or 'top' in args.embType) else 1,
									   args.bs, args.embType, None, None)
	# train initial probe
	allProbes = {}
	# get judge
	judgeM, judgeF = myUtil.loadJudge(args.judge)
	# model extraction starts
	allPosScore = []
	for i in range(args.maxIter):
		# train
		probes = ProbeManager.train(hdManager, args.linearC, args.normReg)
		print(f"\nIter {i + 1}")
		
		# validation
		valCompletion = []
		posScore = 0
		if args.val:
			print(f"Validation Target Prob.: {ProbeManager.getTargetProb(probes, args.evalPT)}")
			# hook model
			hooks = ProbeManager.hookModel(model, probes, args.evalPT)
			valCompletion, allRes = myUtil.GenAndEvalLVLM(model, processor, judgeF, harmValPrompts, args.trainL, args.bs, None, False)
			posScore = torch.tensor(allRes).float().mean().item()
			# unhook model
			for hook in hooks:
				hook.remove()
		# sample
		print(f"Sample Target Prob.: {ProbeManager.getTargetProb(probes, args.pt)}")
		# hook model
		hooks = ProbeManager.hookModel(model, probes, args.pt)
		# get emb
		hdManager, trainCompletion, _ = myUtil.getLVLMEmb(model, hdManager, judgeF, args.thres, layerIdxs,
														 harmTrainPrompts, processor,
														 args.trainL, args.bs,
														 args.embType, hooks, None)  # get steered hd
		posScore = _ if not args.val else posScore
		# unhook model
		for hook in hooks:
			hook.remove()
		allPosScore.append(posScore)
		print(f'History: {sorted(allPosScore)}')
		# save
		allProbes[i] = (posScore, probes, trainCompletion, valCompletion)
		torch.save(allProbes, clfP)
	