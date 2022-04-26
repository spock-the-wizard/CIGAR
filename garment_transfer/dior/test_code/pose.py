import os,sys,json

DIOR_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
 #os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(DIOR_DIR))

from ..models.dior_model import DIORModel
from ..utils.custom_utils import imsave

import torch
import numpy as np
import matplotlib.pyplot as plt

base = DIOR_DIR #'./garment_transfer/dior/'
dataroot = '../data/00_test/raw'#'/shared/rsaas/aiyucui2/inshop/fashion_yifang'
exp_name = 'DIOR_64' # DIORv1_64
epoch = 'latest'
netG = 'dior' # diorv1
ngf = 64

## this is a dummy "argparse" 
class Opt:
    def __init__(self):
        pass
if True:
    opt = Opt()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 8; opt.n_kpts = 18; opt.style_nc = 64
    opt.n_style_blocks = 4; opt.netG = netG; opt.netE = 'adgan'
    opt.ngf = ngf
    opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'; opt.init_gain = 0.02; opt.gpu_ids = [0]
    opt.frozen_flownet = True; opt.random_rate = 1; opt.perturb = False; opt.warmup=False
    opt.name = exp_name
    opt.vgg_path = ''; opt.flownet_path = '' # os.path.join(base, 'checkpoints/DIORv1_64/latest_net_Flow.pth')#'pretrained_models/flownet.pt'
    opt.checkpoints_dir = os.path.join(base, 'checkpoints')
    opt.frozen_enc = True
    opt.load_iter = 0
    opt.epoch = epoch
    opt.verbose = False

# create model
model = DIORModel(opt)
model.setup(opt)

# load data
# from ..datasets.deepfashion_datasets import DFVisualDataset
# Dataset = DFVisualDataset
# ds = Dataset(dataroot=dataroot, dim=(256,176), n_human_part=8)

# # preload a set of pre-selected models defined in "standard_test_anns.txt" for quick visualizations 
# inputs = dict()
# for attr in ds.attr_keys:
#     inputs[attr] = ds.get_attr_visual_input(attr)
    
# define some tool functions for I/O
def load_img(pid, ds):
    if isinstance(pid,str): # load pose from scratch
        return None, None, load_pose_from_json(pid)
    if len(pid[0]) < 10: # load pre-selected models
        person = inputs[pid[0]]
        # print(person)
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
        pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
    else: # load model from scratch
        import pdb;pdb.set_trace()
        person = ds.get_inputs_by_key(pid[0])
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

def load_pose_from_json(ani_pose_dir):
    with open(ani_pose_dir, 'r') as f:
        anno = json.load(f)
    len(anno['people'][0]['pose_keypoints_2d'])
    anno = list(anno['people'][0]['pose_keypoints_2d'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])

    coord = np.concatenate([x[:,None], y[:,None]], -1)
    #import pdb; pdb.set_trace()
    #coord = (coord * 1.1) - np.array([10,30])[None, :]
    pose  = pose_utils.cords_to_map(coord, (256,256), (256, 176))
    pose = np.transpose(pose,(2, 0, 1))
    pose = torch.Tensor(pose)
    return pose

def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    # import pdb;pdb.set_trace()
    if pose != None:
        # import pdb;pdb.set_trace()
        from ..utils import pose_utils #import ..utils.pose_utils as pose_utils
        # print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        # fixed this part so that keypoint image would be normalized 0..1
        kpt = kpt[0]/255
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = (gen_img+1)/2
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        # import pdb;pdb.set_trace()
        # out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])
        if pose != None:
            out = np.concatenate((kpt, out),1)
    else:
        out = kpt
    fig = plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
    plt.axis('off')
    return out

# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2], perturb=False):
    PID = [0,4,6,7]
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    # print(torch.max(pimg),torch.min(pimg))
    if perturb:
        pimg = perturb_images(pimg[None])[0]
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
   
    
    # swap base garment if any
    gimgs = []
    for gid in gids:
        _,_,k = gid
        gimg, gparse, pose =  load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        import pdb;pdb.set_trace()
        # override garments
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2])]
        

    # encode garment (overlay)
    garments = []
    over_gsegs = []
    oimgs = []
    for gid in ogids:
        oimg, oparse, pose = load_img(gid, ds)
        oimgs += [oimg * (oparse == gid[2])]
        seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
        over_gsegs += [seg]
    
    gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, oimgs, gen_img[0], to_pose

if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    for _pid in range(4):
        for _pose_id in range(4):
            # person id
            pid = ("print",_pid, None) # load the 0-th person from "print" group, 
            pose_id = ("print",_pose_id, None) # load the 2-nd person from "print" group, NONE (no) garment is interested
            # generate
            pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, pose_id=pose_id)


            res = plot_img(pimg, gimgs, oimgs, gen_img, pose)
            # import pdb;pdb.set_trace()
            imsave(res,fname='pid%dpose%d.png'%(_pid,_pose_id),fdir='../../data/results/pose')
