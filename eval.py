import argparse
import csv
import json
import os

import torch
from datasets import disable_caching

import ProbeManager
import myUtil

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help="The repo name on huggingface")
	parser.add_argument('--tokenizer', type=str, help="Some repos do not provide tokenizer. So you have to give the repo that contain the tokenizer")
	parser.add_argument('--evalPT', type=str, help="If you want to use the percentile to determine target logits, type in a float [0, 1] (e.g., \"1.0\"). If you want to specify the target logits, type in \"abs[float]\" (e.g., \"abs0\")")
	parser.add_argument('--csvP', type=str, help="The path to store results")
	parser.add_argument('--clfP', type=str, default='None', help="The path to probes")
	parser.add_argument('--evalClfr', type=str, choices=['all', 'first', 'last', 'best'], help="Which probe to use? We recommend \'best\'")
	parser.add_argument('--maxL', type=int, required=True, help="The max new token for generation")
	parser.add_argument('--bs', type=int, required=True, help="Batch size")
	parser.add_argument('--doSample', action='store_true', help="Whether to do sample")
	parser.add_argument('--answerOnly', action='store_true', help="Whether to discard CoT")
	parser.add_argument('--trust', action='store_true', help="Trust remote code?")
	parser.add_argument('--et', action='store_true', help="Thinking?")
	parser.add_argument('--judge', type=str, nargs='+', help="Judges to use")
	parser.add_argument('--evalData', type=str, required=True, help="Which benchmark to use")
	args = parser.parse_args()
	print(args)
	if args.tokenizer is None:
		args.tokenizer = args.model
	# load model & processor
	disable_caching()
	prompts = myUtil.loadData(args.evalData)
	headerLine = ['Dataset', 'Model', 'Sample', 'evalPT', 'ClfP', 'evalClfr', 'maxL', 'answerOnly', 'Thinking']
	valueLine = [args.evalData, args.model, args.doSample, args.evalPT, os.path.split(args.clfP)[-1], args.evalClfr, args.maxL, args.answerOnly, args.et]
	res = []
	with torch.no_grad():
		model, processor, config = myUtil.loadModel(args.model, args.tokenizer, args.trust)
		probeName = 'None'
		if args.clfP != 'None':
			allProbes = torch.load(args.clfP, map_location='cpu', weights_only=False)
			probes, probeName = ProbeManager.getProbe(allProbes, args.evalClfr)
			hooks = ProbeManager.hookModel(model, probes, args.evalPT)
		print(probeName)
		# allComp = myUtil.continuousGen(model, processor, prompts, args.maxL, args.doSample, myUtil.model2thinkend.get(args.model, None) if args.answerOnly else None, rawCompletion=False, enableThink=args.et)
		allComp = myUtil.gen(model, processor, prompts, args.maxL, args.bs, args.doSample, myUtil.model2thinkend.get(args.model, None) if args.answerOnly else None, rawCompletion=False, enableThink=args.et)
		del model
		torch.cuda.empty_cache()
		allScores = myUtil.eval(prompts, allComp, args.judge, args.bs)
		
		k2vs = []
		for i in range(len(prompts)):
			k2v = {}
			for k in allScores.keys():
				k2v[k] = allScores[k][i]
			k2vs.append(k2v)
		res = [{'prompt': p, 'response': c, 'score': r} for p, c, r in zip(prompts, allComp, k2vs)]
		with open("checkOutput.json", "w", encoding="utf-8") as f:
			json.dump(res, f, indent=4, ensure_ascii=False)
		
		for k, v in allScores.items():
			headerLine.append(probeName + ';' + k)
			valueLine.append(torch.tensor(v).float().mean().item())
		with open(args.csvP.replace('.csv', f'_{args.evalPT}.csv'), 'a+', newline='') as f:
			csv.writer(f).writerows([headerLine, valueLine])
		if args.clfP != 'None':
			torch.save(allProbes, args.clfP)
