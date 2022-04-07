import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.dior_model import DIORModel
from utils.custom_utils import imsave
from test_code.pose import dress_in_order, plot_img
from test_code.pose import *


# define dressing-in-order function (the pipeline)
def dress_in_order_texshape(model, pid, target=5, shape_id=None, tex_patch=None, pose_id=None, gids=[], ogids=[], order=[5,1,3,2], perturb=False):
    PID = [0,4,6,7] # which parts to keep?
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)

    if perturb:
        pimg = perturb_images(pimg[None])[0]
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)

    # encode pose? or person?
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)
    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
    
    # checkout shape of gsegs... one for each garment segemnt?
    # import pdb;pdb.set_trace()
    #################################
    # added this part for texshape
    fmap, mask = gsegs[target] # target is the specific segment
    gimg = pimg*(parse==target)
    if shape_id != None:
        gimg, gparse, pose =  load_img(shape_id, ds)
        _, mask = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=target)
        shape_img = [gimg*(gparse==target)]
    else:
        shape_img = []
    if tex_patch != None:
        # replace with fmap of the texture
        fmap = model.netE_attr(tex_patch, model.netVGG)
        
    gsegs[target] = fmap, mask
    gsegs = [gsegs[i] for i in order]
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    # import pdb;pdb.set_trace()
    # shape check
    return pimg, [gimg], shape_img, gen_img[0], to_pose
    
    # #################################

    # # swap base garment if any
    # gimgs = []
    # for gid in gids:
    #     _,_,k = gid
    #     gimg, gparse, pose =  load_img(gid, ds)
    #     seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
    #     gsegs[gid[2]] = seg
    #     gimgs += [gimg * (gparse == gid[2])]

    # # encode garment (overlay)
    # garments = []
    # over_gsegs = []
    # oimgs = []
    # for gid in ogids:
    #     oimg, oparse, pose = load_img(gid, ds)
    #     oimgs += [oimg * (oparse == gid[2])]
    #     seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
    #     over_gsegs += [seg]
    
    # gsegs = [gsegs[i] for i in order] + over_gsegs
    # gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, oimgs, gen_img[0], to_pose


if __name__=='__main__':
    # for i in range(8):
    for a in range(4):
        for b in range(4):
            
            i = 5
            pid = ('strip', a, i) 
            patch_id = ('collar', b, i) 
            patch, parse, from_pose = load_img(patch_id, ds)
            # imsave(parse,fdir='../data/results/texshape',prefix='justtexture')
            # import pdb;pdb.set_trace()
            pimg, gimgs, oimgs, gen_img, pose = dress_in_order_texshape(model, pid, target=5,tex_patch=patch[None])
            res = plot_img(pimg, [patch], oimgs, gen_img, pose)
            imsave(res,fname='texshape_strip%d_collar%d.png' % (a,b),fdir='../data/results/texshape')

