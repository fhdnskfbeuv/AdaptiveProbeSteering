import torch


class HDManager:
	def __init__(self, layerIdxs):
		self.layer2hd = {i: {'hd': None, 'label': None} for i in layerIdxs}
	
	def add(self, layerIdx, hd, label):  # [B, D], [B, ]
		self.layer2hd[layerIdx]['hd'] = torch.concat([self.layer2hd[layerIdx]['hd'], hd], dim=0) if self.layer2hd[layerIdx]['hd'] is not None else hd
		self.layer2hd[layerIdx]['label'] = torch.concat([self.layer2hd[layerIdx]['label'], torch.tensor(label, dtype=torch.long)], dim=0) if self.layer2hd[layerIdx]['label'] is not None else torch.tensor(label, dtype=torch.long)
		
	def save(self, savePath):
		torch.save(self.layer2hd, savePath)
		
	def load(self, loadPath):
		self.layer2hd = torch.load(loadPath, map_location='cpu')

		


