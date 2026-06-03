import copy
from functools import partial

import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression as skLR

import HiddenStateManager
from cuml.linear_model import LogisticRegression
from cuml.svm.linear_svc import LinearSVC


class ProbeManager:
	def __init__(self):
		self.probes = {}
	
	def getLayerIdxs(self):
		return self.probes.keys()
	
	@torch.no_grad()
	def train(self, hdManager: HiddenStateManager.HDManager, linearC, normReg: bool):
		self.probes = {}
		for layerIdx, batch in hdManager.layer2hd.items():
			x = batch['hd'].clone().detach().numpy()
			C = min(1.0, 1 / torch.norm(batch['hd'].float(), dim=-1).mean().item()) if normReg else 1.0
			y = batch['label'].clone().detach().numpy()
			totalWeight = x.shape[0]
			sampleWeight = np.ones_like(y, dtype=np.float32)
			posClassWeight = totalWeight / (2 * sampleWeight[y == 1].sum().item())
			negClassWeight = totalWeight / (2 * sampleWeight[y == 0].sum().item())
			sampleWeight[y == 1] *= posClassWeight
			sampleWeight[y == 0] *= negClassWeight
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
			score = (batch['hd'][y == 1, :].cuda().float() @ w.T.cuda().float() + b.cuda().float()).squeeze(-1)  # (B, )
			self.probes[layerIdx] = {'w': w, 'b': b, 'score': score}  # (1, D), (1, ), (B, )
	
	def getTargetScores(self, strength: str):
		ret = {}
		for layerIdx, probe in self.probes.items():
			if strength == 'mean':
				ret[layerIdx] = torch.mean(probe['score'])
			else:
				ret[layerIdx] = torch.quantile(probe['score'], float(strength)) if 'abs' not in strength else torch.tensor(float(strength.replace('abs', '')))
		return ret
	
	@torch.no_grad()
	def getTargetProb(self, strength: str):
		ret = {}
		for layerIdx, probe in self.probes.items():
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


def hookModel(model, pm: ProbeManager, layerIdxs, strength):
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	hooks = []
	strengths = pm.getTargetScores(strength) if isinstance(strength, str) else copy.deepcopy(strength)
	switch = Switch(enable=True)
	for i in layerIdxs:
		whateverPara = next(baseModel.layers[i].mlp.named_parameters())[1]
		wNorm = torch.norm(pm.probes[i]['w'], dim=-1).to(whateverPara)
		wNorm[wNorm == 0.0] = 1e-6
		if isinstance(pm.probes[i]['b'], float):
			pm.probes[i]['b'] = torch.tensor(pm.probes[i]['b'])
		hook = baseModel.layers[i].register_forward_hook(
			partial(
				probeSteer,
				w=pm.probes[i]['w'].to(whateverPara),
				b=pm.probes[i]['b'].to(whateverPara),
				s=strengths[i].to(whateverPara),
				wNorm=wNorm.to(whateverPara),
				switch=switch
			)
		)
		baseModel.layers[i]._forward_hooks.move_to_end(hook.id, last=False)  # my hook comes first
		hooks.append(hook)
	hooks.append(switch)
	return hooks
