import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

SAVE_DIR = '../data/results'

def imsave(img,fname=None,prefix='test',fdir=None):
    
    if fdir==None:
        fdir = SAVE_DIR

    if fname==None:
        idx = len([f for f in os.listdir(fdir) if f.startswith(prefix)])
        
        fname = '%s%d.png'%(prefix,idx)
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    pth = os.path.join(fdir,fname)
    # import pdb;pdb.set_trace()
    if isinstance(img,np.ndarray):
        plt.imsave(pth,img)
    else:
        save_image(img,pth)