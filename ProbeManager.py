from peft import PeftModel
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.svm import LinearSVC
from cuml.linear_model import LogisticRegression
import HiddenStateManager
import torch
import numpy as np


class SCAVAdapter(torch.nn.Module):
	def __init__(self, hiddenSize: int) -> None:
		super().__init__()
		self.downProj = torch.nn.Linear(hiddenSize, 1, bias=True, dtype=torch.float32)
		self.act = torch.nn.ReLU()
		self.upProj = torch.nn.Linear(1, hiddenSize, bias=False, dtype=torch.float32)
	
	def forward(self, inputHD: torch.Tensor):
		return inputHD + self.upProj(self.act(self.downProj(inputHD)))


class MySequential(torch.nn.Module):
	def __init__(self, allModule):
		super().__init__()
		self.allModule = allModule
	
	def forward(self, hiddenStates, **inputs):
		output = self.allModule[0](hiddenStates, **inputs)
		for i in range(1, len(self.allModule)):
			output = self.allModule[i](output)
		# print(output)
		return output


def wrapModel(model, mlps, layerIdxs):
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	wrappedLayers = []
	for i in range(len(baseModel.layers)):
		if i in layerIdxs:
			whateverPara = next(baseModel.layers[i].named_parameters())[1]
			wrappedLayers.append(MySequential([baseModel.layers[i], mlps[i].to(whateverPara.dtype).to(whateverPara.device)]))
		else:
			wrappedLayers.append(baseModel.layers[i])
	baseModel.layers = torch.nn.ModuleList(wrappedLayers)
	return model


def unwrapModel(model):
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	unwrappedLayers = []
	for i in range(len(baseModel.layers)):
		if isinstance(baseModel.layers[i], MySequential):
			unwrappedLayers.append(baseModel.layers[i].allModule[0])
			del baseModel.layers[i].allModule[1]
		else:
			unwrappedLayers.append(baseModel.layers[i])
	baseModel.layers = torch.nn.ModuleList(unwrappedLayers)
	return model


class ProbeManager:
	def __init__(self):
		self.probes = {}
		
	def getLayerIdxs(self):
		return self.probes.keys()
	
	@torch.no_grad()
	def train(self, hdManager: HiddenStateManager.HDManager, useGPU: bool, normReg: bool):
		self.probes = {}
		for layerIdx, batch in hdManager.layer2hd.items():
			x = batch['hd'].clone().numpy()
			C = 1.0 / torch.norm(batch['hd'].float(), dim=-1).mean().item() if normReg else 1.0
			y = batch['label'].clone().numpy()
			totalWeight = x.shape[0]
			posClassWeight = totalWeight / (2 * (y == 1).sum().item())
			negClassWeight = totalWeight / (2 * (y == 0).sum().item())
			sampleWeight = np.ones_like(y, dtype=np.float32)
			sampleWeight[y == 1] *= posClassWeight
			sampleWeight[y == 0] *= negClassWeight
			if useGPU:
				linear = LogisticRegression(solver="qn", max_iter=10000,
											# class_weight='balanced',
											output_type='numpy', penalty='l2', C=C, fit_intercept=True)
				linear.verbose = 0
			else:
				linear = skLR(solver="saga", max_iter=10000,
							  # class_weight='balanced',
							  warm_start=True,
							  n_jobs=16, penalty='l2', C=C, fit_intercept=True)
				linear.verbose = 0
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
	
	@torch.no_grad()
	def toMLP(self, strength: str):
		mlps = {}
		allScores = self.getTargetScores(strength)
		for layerIdx, probe in self.probes.items():
			targetScore = allScores[layerIdx]
			mlps[layerIdx] = SCAVAdapter(probe['w'].shape[1])
			wNorm = torch.norm(probe['w'], dim=-1).item()
			if wNorm == 0.0:
				wNorm += 1e-6
			mlps[layerIdx].downProj.weight.copy_(-probe['w'] / wNorm)
			mlps[layerIdx].upProj.weight.copy_(probe['w'].T / wNorm)
			mlps[layerIdx].downProj.bias.copy_((targetScore - probe['b']) / wNorm)
		return mlps


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
