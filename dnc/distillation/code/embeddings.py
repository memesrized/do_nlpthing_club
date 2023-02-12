import torch
import numpy as np
from navec import Navec


class CustomEmbeddings:
    def __init__(self, path, unk="<unk>"):
        self.emb = Navec.load(path)
        self.unk = unk
    
    def _convert(self, list_):
        """UserWarning: Creating a tensor from a list
        of numpy.ndarrays is extremely slow. 
        
        Please consider converting the list to a single
        numpy.ndarray with numpy.array() before converting to a tensor."""
        return torch.as_tensor(np.array(list_))
    
    def __call__(self, tokens):
        return self._convert([self.emb.get(x, self.emb[self.unk]) for x in tokens])
