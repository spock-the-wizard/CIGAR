import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.dior_model import DIORModel
from utils.custom_utils import imsave
from test_code.pose import dress_in_order, plot_img
from test_code.pose import *

def save_reference_images(ds=ds):
    for k,lst in inputs.items():
        for idx,img in enumerate(lst):
            pid = (k,idx,None)
            res = load_img(pid,ds)
            imsave(res[0],fname='%s%d.png'%(k,idx),fdir='../data/results/reference')
            # import pdb;pdb.set_trace()

# see what each segment id is
# turns out, they refer to the same segments!
def check_segmentation(ds=ds):
    pid = ("pattern", 3, None)
    for i in range(10):
        gids = [
        ("plaid",1,i),
        ("pattern",3,i)]

        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2,i,1])
        res1 = plot_img(pimg, gimgs)

        imsave(res1,fname='segmentcheck%d.png'%i,fdir='../data/results/order')

if __name__=='__main__':
        
    pid = ("print", 0, None) # load the 3-rd person from "pattern" group, NONE (no) garment is interested
    for i in range(10):
        gids = [
        ("flower",1,i), # load the 0-th person from "plaid" group, garment #5 (top) is interested
        ("pattern",3,i),  # load the 3-rd person from "pattern" group, garment #1 (bottom) is interested
        ]

        # tuck in (dressing order: hair, top, bottom)
        pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2,i,1])
        res1 = plot_img(pimg, gimgs)
        imsave(res1,fname='segmentcheck_flower1_%d.png'%i,fdir='../data/results/order')
        # break
    
    # # not tuckin (dressing order: hair, bottom, top)
    # pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2,1,5])
    # res2 = plot_img(pimg, gimgs, gen_img=gen_img, pose=pose)