"""Check optimizer continuation and label/weight semantics before real fitting."""
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from expansion_softmax_readout import fit_softmax

def main():
    torch.set_num_threads(2);torch.manual_seed(73)
    ids=np.array([2,5,9]);y=np.repeat(ids,[18,9,5])
    centers=torch.eye(3,8);E=F.normalize(centers[np.searchsorted(ids,y)]+.1*torch.randn(len(y),8),dim=1)
    with tempfile.TemporaryDirectory(prefix='softmax-head-test-') as tmp:
        h=Path(tmp)
        full,report=fit_softmax(E,y,ids,1e-3,h/'full.pt',max_steps=100)
        fit_softmax(E,y,ids,1e-3,h/'resume.pt',max_steps=50)
        resumed,_=fit_softmax(E,y,ids,1e-3,h/'resume.pt',max_steps=100)
        assert torch.equal(full,resumed),'Optimizer resume changed weights.'
        pred=ids[(E@full[:-1]+full[-1]).argmax(1).numpy()]
        assert (pred==y).all(),'Class-ID mapping or fit broken.'
        assert report['losses'][-1][1]<.2*report['initial_loss']
        print('PASS: exact optimizer resume, imbalanced class fit, class-ID mapping, finite loss decrease.')
if __name__=='__main__':main()
