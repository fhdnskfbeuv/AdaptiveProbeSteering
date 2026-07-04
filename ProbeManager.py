import copy
from functools import partial

import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression as skLR
import HiddenStateManager
from cuml.linear_model import LogisticRegression
from cuml.svm.linear_svc import LinearSVC


@torch.no_grad()
def train(hdManager: HiddenStateManager.HDManager, linearC, normReg: bool, tb: bool):
	probes = {}
	for layerIdx, batch in hdManager.layer2hd.items():
		x = batch['hd'].clone().detach().numpy()
		C = (1 / (torch.norm(batch['hd'].float(), dim=-1).mean().item() ** 2)) if normReg else 1.0
		y = batch['label'].clone().detach().numpy()
		totalWeight = x.shape[0]
		sampleWeight = np.ones_like(y, dtype=np.float32)
		
		posIndex = y >= 0  # if np.unique(y[y >= 0]).shape[0] == 1 else (y > 0)
		negIndex = y < 0  # if np.unique(y[y >= 0]).shape[0] == 1 else (y <= 0)

		# Class Balance
		posClassWeight = totalWeight / (2 * sampleWeight[posIndex].sum().item())
		negClassWeight = totalWeight / (2 * sampleWeight[negIndex].sum().item())
		sampleWeight[posIndex] *= posClassWeight
		sampleWeight[negIndex] *= negClassWeight
		if tb:
			# Topic Balance
			uniquePos = np.unique(y[posIndex])
			uniqueNeg = np.unique(y[negIndex])
			sliceNum = uniqueNeg.shape[0]
			absY = np.abs(y)
			incelIndexes = []
			for un in uniqueNeg:
				if abs(un) in uniquePos:  # chad
					oriWeightSum = np.sum(sampleWeight[absY == abs(un)]).item()
					sampleWeight[absY == abs(un)] *= totalWeight / (sliceNum * oriWeightSum) if oriWeightSum != 0 else 0
					sliceWeightSum = np.sum(sampleWeight[absY == abs(un)]).item()
					sampleWeight[y == un] *= sliceWeightSum / (2 * np.sum(sampleWeight[y == un]).item())
					sampleWeight[y == (-un)] *= sliceWeightSum / (2 * np.sum(sampleWeight[y == (-un)]).item())
				else:  # incel
					incelIndexes.append(un)
			incelWeight = totalWeight * len(incelIndexes) / sliceNum
			for incelIndex in incelIndexes:
				sampleWeight[y == incelIndex] *= incelWeight / (2 * len(incelIndexes) * np.sum(sampleWeight[y == incelIndex]))
			sampleWeight[y == 0] *= incelWeight / (2 * np.sum(sampleWeight[y == 0]))
		y[posIndex] = 1
		y[negIndex] = 0
		if linearC == 'cuLR':
			linear = LogisticRegression(solver="qn", max_iter=10000,
										# class_weight='balanced',
										output_type='numpy', penalty='l2', C=C, fit_intercept=True)
			linear.verbose = 0
		elif linearC == 'cuSVC':
			linear = LinearSVC(output_type='numpy', penalty='l2', C=C, fit_intercept=True)
			linear.verbose = 0
		elif linearC == 'skLR':
			linear = skLR(solver="saga", max_iter=10000,
						  # class_weight='balanced',
						  warm_start=True,
						  n_jobs=16, penalty='l2', C=C, fit_intercept=True)
			linear.verbose = 0
		else:
			print(f'{linearC} not implemented')
			exit(1)
		linear.fit(x, y, sample_weight=sampleWeight)
		w = torch.tensor(linear.coef_, requires_grad=False)
		b = torch.tensor(linear.intercept_, requires_grad=False)
		selectedIndex = (y == 1) * (sampleWeight != 0)
		score = (batch['hd'][selectedIndex, :].cuda().float() @ w.T.cuda().float() + b.cuda().float()).squeeze(-1)  # (B, )
		probes[layerIdx] = {'w': w, 'b': b, 'score': score}  # (1, D), (1, ), (B, )
	return probes

@torch.no_grad()
def getTargetScores(probes, strength: str):
	ret = {}
	for layerIdx, probe in probes.items():
		if strength == 'mean':
			ret[layerIdx] = torch.mean(probe['score'])
		else:
			ret[layerIdx] = torch.quantile(probe['score'], float(strength)) if 'abs' not in strength else torch.tensor(float(strength.replace('abs', '')))
	return ret


@torch.no_grad()
def getTargetProb(probes, strength: str):
	ret = {}
	for layerIdx, probe in probes.items():
		if strength == 'mean':
			ret[layerIdx] = torch.sigmoid(torch.mean(probe['score'])).item()
		else:
			ret[layerIdx] = torch.sigmoid(torch.quantile(probe['score'], float(strength))).item() if 'abs' not in strength else torch.sigmoid(torch.tensor(float(strength.replace('abs', '')))).item()
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
	(iterNum, (score, probes, trainCompletion, valCompletion)) = list(probe2test.items())[0]
	return probes, f'Iter{iterNum}, {score}'


class Switch:
	def __init__(self, enable):
		self.enable: bool = enable
	
	def remove(self):
		pass


def probeSteer(module, inputs, outputs, w, b, s, wNorm, switch: Switch):  # [B, L, D], [1, D], [1, ], [1, ], [1, ]
	if switch.enable:
		return outputs + (torch.relu(s - outputs @ w.T - b) / wNorm) @ (w / wNorm)
	return outputs


def hookModel(model, pm, strength):
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	hooks = []
	strengths = getTargetScores(pm, strength) if isinstance(strength, str) else copy.deepcopy(strength)
	switch = Switch(enable=True)
	for i in pm.keys():
		whateverPara = next(baseModel.layers[i].mlp.named_parameters())[1]
		wNorm = torch.norm(pm[i]['w'], dim=-1).to(whateverPara)
		wNorm[wNorm == 0.0] = 1e-6
		if isinstance(pm[i]['b'], float):
			pm[i]['b'] = torch.tensor(pm[i]['b'])
		hook = baseModel.layers[i].register_forward_hook(
			partial(
				probeSteer,
				w=pm[i]['w'].to(whateverPara),
				b=pm[i]['b'].to(whateverPara),
				s=strengths[i].to(whateverPara),
				wNorm=wNorm.to(whateverPara),
				switch=switch
			)
		)
		baseModel.layers[i]._forward_hooks.move_to_end(hook.id, last=False)  # my hook comes first
		hooks.append(hook)
	hooks.append(switch)
	return hooks
