import argparse
import os
import shutil

from datasets import disable_caching

import HiddenStateManager
import myUtil
import ProbeManager
import torch

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help="The repo name on huggingface")
	parser.add_argument('--tokenizer', type=str, help="Some repos do not provide tokenizer. So you have to give the repo that contain the tokenizer")
	parser.add_argument('--pt', type=str, required=True, help="If you want to use the percentile to determine target logits, type in a float [0, 1] (e.g., \"1.0\"). If you want to specify the target logits, type in \"abs[float]\" (e.g., \"abs0\")")
	parser.add_argument('--evalPT', type=str, help="If you want to use the percentile to determine target logits, type in a float [0, 1] (e.g., \"1.0\"). If you want to specify the target logits, type in \"abs[float]\" (e.g., \"abs0\")")
	parser.add_argument('--thres', type=float, required=True, nargs='+', help="The low and the high threshold for the annotator (e.g., 0.05 0.6)")
	parser.add_argument('--maxIter', type=int, required=True, help="The maximum iteration of adaptive retraining")
	parser.add_argument('--trainL', type=int, required=True, help="the max new token during adaptive retraining")
	parser.add_argument('--embType', type=str, required=True, choices=["last", "response", "all", "prompt"])
	parser.add_argument('--saveDir', type=str, required=True, help="The root of probe storage")
	parser.add_argument('--judge', type=str, required=True, help="The annotator")
	parser.add_argument('--layer', nargs='+', type=float, default=[-2], help="The interval of selected layer. Length <= 2")
	parser.add_argument('--linearC', type=str, choices=['cuLR', 'cuSVC', 'skLR'], help="The type of linear model")
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
	model, processor, config = myUtil.loadModel(args.model, args.tokenizer)
	saveDir = os.path.join(args.saveDir, args.model.replace('./', '').replace('/', '_'))
	os.makedirs(saveDir, exist_ok=True)
	judgeN = args.judge.split(' ')[0]
	layerIdxs = myUtil.getLayer(config.num_hidden_layers, args.layer)
	clfP = os.path.join(saveDir,
						f'judge{judgeN}_embType{args.embType}_filter{args.filter}_normReg{args.normReg}_layer[{layerIdxs[0]}, {layerIdxs[-1]}]_linearC{args.linearC}_maxIter{args.maxIter}_trainL{args.trainL}_pt{args.pt}_softThres{args.thres}.pt'.replace(
							'/',
							'-'))
	harmTrainPrompts, benignTrainPrompts, harmValPrompts, _ = myUtil.loadData('train')
	if args.filter:
		benignTrainPrompts = myUtil.filterData(model, processor, args.judge, benignTrainPrompts, 512, args.thres[1], None)
	# get initial embd
	hdManager = HiddenStateManager.HDManager(layerIdxs)
	# get the hd you don't prefer
	hdManager, _, _ = myUtil.getLLMEmb(model, hdManager, lambda x, y: 0.0, args.thres, layerIdxs,
									   harmTrainPrompts, processor,
									   args.trainL if (args.embType in ['all', 'response']) else 1,
									   args.embType, None)
	# get the hd you prefer
	hdManager, _, _ = myUtil.getLLMEmb(model, hdManager, lambda x, y: 1.0, args.thres, layerIdxs,
									   benignTrainPrompts, processor,
									   args.trainL if (args.embType in ['all', 'response']) else 1,
									   args.embType, None)
	# train initial probe
	allProbes = {}
	# get judge
	judgeM, judgeF = myUtil.loadJudge(args.judge)
	# model extraction starts
	allPosScore = []
	for i in range(args.maxIter):
		# train
		probes = ProbeManager.ProbeManager()
		probes.train(hdManager, args.linearC, args.normReg)
		print(f"\nIter {i + 1}")
		
		# validation
		valCompletion = []
		posScore = 0
		if args.val:
			print(f"Validation Target Prob.: {probes.getTargetProb(args.evalPT)}")
			# hook model
			hooks = ProbeManager.hookModel(model, probes, probes.getLayerIdxs(), args.evalPT)
			valCompletion, allRes = myUtil.GenAndEval(model, processor, judgeF, harmValPrompts, args.trainL, False)
			posScore = torch.tensor(allRes).float().mean().item()
			# unhook model
			for hook in hooks:
				hook.remove()
		
		# sample
		print(f"Sample Target Prob.: {probes.getTargetProb(args.pt if args.pt != 'adaptive' else str(posScore))}")
		# hook model
		hooks = ProbeManager.hookModel(model, probes, probes.getLayerIdxs(), args.pt)
		# get emb
		hdManager, trainCompletion, _ = myUtil.getLLMEmb(model, hdManager, judgeF, args.thres, layerIdxs,
														 harmTrainPrompts, processor,
														 args.trainL,
														 args.embType, None)  # get steered hd
		posScore = _ if not args.val else posScore
		# unhook model
		for hook in hooks:
			hook.remove()
		allPosScore.append(posScore)
		print(f'History: {sorted(allPosScore)}')
		# save
		allProbes[i] = (posScore, probes, trainCompletion, valCompletion)
		torch.save(allProbes, clfP)
	
	del judgeM
	torch.save(allProbes, clfP)
