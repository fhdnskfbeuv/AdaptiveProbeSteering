import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression as skLR

import HiddenStateManager
from cuml.linear_model import LogisticRegression
from cuml.svm.linear_svc import LinearSVC
from functools import partial


class ProbeManager:
	def __init__(self):
		self.probes = {}
	
	def getLayerIdxs(self):
		return self.probes.keys()
	
	@torch.no_grad()
	def train(self, hdManager: HiddenStateManager.HDManager, linearC, normReg: bool):
		self.probes = {}
		for layerIdx, batch in hdManager.layer2hd.items():
			x = batch['hd'].clone().numpy()
			C = min(1.0, 1.0 / torch.norm(batch['hd'].float(), dim=-1).mean().item()) if normReg else 1.0
			y = batch['label'].clone().numpy()
			totalWeight = x.shape[0]
			sampleWeight = np.ones_like(y, dtype=np.float32)
			posClassWeight = totalWeight / (2 * sampleWeight[y == 1].sum().item())
			negClassWeight = totalWeight / (2 * sampleWeight[y == 0].sum().item())
			sampleWeight[y == 1] *= posClassWeight
			sampleWeight[y == 0] *= negClassWeight
			print(f'pos: {(y == 1).sum().item()}; neg: {(y == 0).sum().item()}')
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
			self.probes[layerIdx] = {'w': w, 'b': b.item(), 'score': score}  # (1, D), scaler, (B, )
	
	def getTargetScores(self, strength: str):
		ret = {}
		for layerIdx, probe in self.probes.items():
			ret[layerIdx] = torch.quantile(probe['score'], float(strength)).item() if 'abs' not in strength else float(strength.replace('abs', ''))
		return ret
	
	@torch.no_grad()
	def getTargetProb(self, strength: str):
		ret = {}
		for layerIdx, probe in self.probes.items():
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


def probeSteer(module, inputs, hd, w, b, s, wNorm):  # [B, L, D], [1, D], scalar, scalar, scalar
	return hd + (torch.relu(s - hd @ w.T - b) / wNorm) @ (w / wNorm)


def hookModel(model, pm: ProbeManager, layerIdxs, strength: str):
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	hooks = []
	strengths = pm.getTargetScores(strength)
	for i in layerIdxs:
		whateverPara = next(baseModel.layers[i].mlp.named_parameters())[1]
		wNorm = torch.norm(pm.probes[i]['w'], dim=-1).item()
		if wNorm == 0.0:
			wNorm += 1e-6
		hook = baseModel.layers[i].register_forward_hook(
			partial(
				probeSteer,
				w=pm.probes[i]['w'].to(whateverPara),
				b=pm.probes[i]['b'],
				s=strengths[i],
				wNorm=wNorm
			)
		)
		hooks.append(hook)
	return hooks
