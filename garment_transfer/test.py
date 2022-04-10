from dior.test_code.pose import *


def get_texshape_from_garment(model,pid,gid,ds):
    # load garment image
    pimg, pparse, from_pose = load_img(pid,ds)
    gimg, gparse, to_pose = load_img(gid,ds)
    tex,shape = model.encode_single_attr(gimg[None],gparse[None],from_pose[None], to_pose[None],i=gid[2])
    return tex,shape,gimg

if __name__=="__main__":
    TEST_PID = 1

    pid = ("print", TEST_PID, None)
    gids = [
    ("flower",1,5),
    ("pattern",3,2),
    ("pattern",3,5),
    
    ("flower",2,5),
    
    ("print",3,5),
    ("print",2,5),
    ]

    for gid in gids:
        group,idx,gidx = gid

        tex,shape,gimg = get_texshape_from_garment(model,pid,gid,ds)
        imsave(shape[0],fname='%s_%s_%d_mask.png'%(group,idx,gidx),fdir='../data/00_test/results')
        imsave(gimg,fname='%s_%s_%d_gimg.png'%(group,idx,gidx),fdir='../data/00_test/results')
        # import pdb;pdb.set_trace()
