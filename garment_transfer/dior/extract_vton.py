"""Code to Dress up specified garmnet on User image"""

from garment_transfer.dior.test_code.pose import *
from garment_transfer.dior.datasets.deepfashion_datasets import *
from torchvision.utils import save_image
from utils.util import glue_images
from PIL import ImageOps

# ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
#                'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
ORDER = [0,1,2,3,4,10,11,12,5,6,7,13,14,15,8,16,9,17] #16,8,17,9,10,1,2,11,3,12,4,13,5,14,6,15,7]

class IngredientsLibrary():
    def __init__(self,usr_dir,gar_dir,parse_dir,pose_csv,crop_size=(256,176)):
        self.usr_dir = usr_dir
        self.gar_dir = gar_dir
        self.parse_dir = parse_dir
        self.pose_csv = pose_csv
        self.crop_size = crop_size
        # transforms
        self.resize = transforms.Resize(crop_size)
        self.toTensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       
        self.annotation_file = pd.read_csv(pose_csv,sep=':').set_index('name')
    
        self.aiyu2atr, self.atr2aiyu = get_label_map(8)

    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        img = ImageOps.exif_transpose(img)
        self.old_size = np.array(img).shape[:2]
        img = self.resize(img)
        img = self.toTensor(img)
#        img = self.normalize(img)
        
        return img
    
    def _load_mask(self, fn): 
        mask = Image.open(fn + ".png")
        mask = self.resize(mask)
        mask = torch.from_numpy(np.array(mask))
        
        texture_mask = copy.deepcopy(mask)
        for atr in self.atr2aiyu:
            aiyu = self.atr2aiyu[atr]
            texture_mask[mask == atr] = aiyu
        return texture_mask
    
    def _load_kpt(self, name):
        try:
            string = self.annotation_file.loc[str(name)]
            array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
            # import pdb;pdb.set_trace()
            array = array[ORDER] #[array[i] for i in ORDER]
            pose  = pose_utils.cords_to_map(array, self.crop_size, self.old_size)
            # import pdb;pdb.set_trace()

            pose = np.transpose(pose,(2, 0, 1))
            pose = torch.Tensor(pose)
            return pose
        except:
            import pdb;pdb.set_trace()
            return torch.zeros((18,256,176))
    
    def load_item(self, key):
        # import pdb;pdb.set_trace()
        pth_img = os.path.join(self.usr_dir,key)
        if not os.path.exists(pth_img):
            pth_img = os.path.join(self.gar_dir,key)
        img = self._load_img(pth_img)
        kpt = self._load_kpt(key)     
        parse = self._load_mask(os.path.join(self.parse_dir, key[:-4]))
        return img.cuda(), kpt.cuda(), parse.cuda()

    def load_parse(self, fn):
        pth_parse = os.path.join(self.parse_dir, fn[:-4]+'.png')
        parse = Image.open(pth_parse)
        parse = self.resize(parse)
        parse = torch.from_numpy(np.array(parse))
        return parse.cuda()


def dress_up(ds,user,g_keys,g_cats,order=[1,5,3,2],debug=False):
    
    PID=[0,4,6,7]
    
    pimg, from_pose, parse = ds.load_item(user)
    to_pose = from_pose

    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
    
    g_cls = catlabel2index(g_cats)
    # swap base garment if any
    gimgs = []
    for gkey,gcat,gid in zip(g_keys,g_cats,g_cls): # in gids:
        gimg, pose, gparse = ds.load_item(gkey) #garment)load_img(gid, ds)
        
        # import pdb;pdb.set_trace()      
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid)

        # debug_pose = torch.max(pose,axis=0).values
        # save_image(debug_pose,'test_gpose.png')
        # override garments
        gsegs[gid] = seg
        gimgs += [gimg * (gparse == gid)]

        if gcat == 'dress':
                BOTTOM_IDX = 1
                SKIN_IDX = 6
                # seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=1)
                # override pants texture with texture feat from dress
                gsegs[BOTTOM_IDX] = (gsegs[SKIN_IDX][0],gsegs[BOTTOM_IDX][1])
                # order = [1,5]
                # gimgs += [gimg*(gparse==1)]


    # import pdb;pdb.set_trace()
    gsegs = [gsegs[i] for i in order]
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    '''
    import pdb;pdb.set_trace()
    # override face and hair
    parse = ds.load_parse(user)
    fmask = pimg*(parse == 13)
    hmask = pimg*(parse == 2)
    # remove face and hair from gen_img
    eraser = torch.where(parse==13, 0, 1)
    gen_img[0] = gen_img[0] * eraser
    eraser = torch.where(parse==2, 0, 1)
    gen_img[0] = gen_img[0] * eraser
    # override new face from fmask
    gen_img[0] += fmask
    # override hair from hmask
    gen_img[0] += hmask
    '''
    # plt.imsave(res,'shi.png') #import pdb;pdb.set_trace()
    # save_image(res,'shi.png')
    return pimg, gimgs, gen_img[0], to_pose
    
def dress_up_single(ds,users,g_keys,g_cats,order=[1,5,3,2],debug=False):
    
    PID=[0,4,6,7]

    pimgs = []
    poses = []
    parses = []

    for user in users:
        user = user.split('/')[-1]

        pimg, from_pose, parse = ds.load_item(user)

        pimgs.append(pimg[None])
        poses.append(from_pose[None])
        parses.append(parse[None])

    pimgs = torch.cat(pimgs)
    poses = torch.cat(poses)
    parses = torch.cat(parses)

    with torch.no_grad():
        psegs = model.encode_attr(pimgs,parses,poses,poses,PID)
        gsegs = model.encode_attr(pimgs,parses,poses,poses)
        # psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)
        # gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
        
        g_cls = catlabel2index(g_cats)
        # swap base garment if any
        gimgs = []
        for gkey,gcat,gid in zip(g_keys,g_cats,g_cls): # in gids:
            gimg, pose, gparse = ds.load_item(gkey) #garment)load_img(gid, ds)
            
            gimg_ = gimg.expand(len(users),*gimg.shape)  
            gparse_ = gparse.expand(len(users),*gparse.shape)
            gpose_ = pose.expand(len(users),*pose.shape)
            seg = model.encode_single_attr(gimg_, gparse_,gpose_, poses, i=gid)

            # debug_pose = torch.max(pose,axis=0).values
            # save_image(debug_pose,'test_gpose.png')
            # override garments
            gsegs[gid] = seg
            gimgs += [gimg * (gparse == gid)]

            if gcat == 'dress':
                BOTTOM_IDX = 1
                SKIN_IDX = 6
                # seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=1)
                # override pants texture with texture feat from dress
                gsegs[BOTTOM_IDX] = (gsegs[SKIN_IDX][0],gsegs[BOTTOM_IDX][1])
                # order = [1,5]
                # gimgs += [gimg*(gparse==1)]


        gsegs = [gsegs[i] for i in order]
        gen_img = model.netG(poses, psegs, gsegs)

    '''
    import pdb;pdb.set_trace()
    # override face and hair
    parse = ds.load_parse(user)
    fmask = pimg*(parse == 13)
    hmask = pimg*(parse == 2)
    # remove face and hair from gen_img
    eraser = torch.where(parse==13, 0, 1)
    gen_img[0] = gen_img[0] * eraser
    eraser = torch.where(parse==2, 0, 1)
    gen_img[0] = gen_img[0] * eraser
    # override new face from fmask
    gen_img[0] += fmask
    # override hair from hmask
    gen_img[0] += hmask
    '''
    # plt.imsave(res,'shi.png') #import pdb;pdb.set_trace()
    # save_image(res,'shi.png')
    return pimgs, gimgs, gen_img, poses

def catlabel2index(cats):
    ind = []
    cdict = {
        'shirt': 5,
        'toptee': 5,
        'bottom': 1,
        'dress' : 5,
    }
    for label in cats:
        print('@'*50)
        print(label)
        ind.append(cdict[label])
    return ind
        
def extract_vton(users,garments,cats,usr_dir,gar_dir,parse_dir,pose_csv):
    """
        user_pth: pth of user image
        garments: list of paths to garment image
    """
    # import pdb;pdb.set_trace()
    # labels = catlabel2index(cats)
    ds = IngredientsLibrary(usr_dir,gar_dir,parse_dir,pose_csv)
    
    g_pths = [file.split('/')[-1] for file in garments]
    
    debugs = []
    results = []
    user = users[0].split('/')[-1]
    # pimgs,gimgs,vtons,poses = dress_up(ds,users,g_pths,cats,debug=True)
    # single user
    usr,gars,vton,user_pose = dress_up(ds,user,g_pths,cats,debug=True)
    res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose)
    imsave(res,prefix='debug',fdir='/home/ubuntu/efs/CIGAR/data/05_results')
    imsave(res,fname='debug.png',fdir='/home/ubuntu/efs/CIGAR/data/04_vton_results')
    
    imsave(vton,prefix='results',fdir='/home/ubuntu/efs/CIGAR/data/05_results')
    imsave(vton,fname='results.png',fdir='/home/ubuntu/efs/CIGAR/data/04_vton_results')

    # res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose[1])
    # imsave(res,fname='pose1.png',fdir='./')
    # plt.imsave('hi3.png',res) 
    # save_image(res,'hi.png')
    
    # res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose[1])
    return vton

    for pimg,vton,pose in zip(pimgs,vtons,poses):
        res = plot_img(pimg=pimg,gimgs=gimgs,gen_img=vton,pose=pose)
    
        # res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose[0])
        debugs.append(res)
        results.append(vton)

    # save debug image and copy
    debugs = cv2.vconcat(debugs)
    imsave(debugs,prefix='debug',fdir='/home/ubuntu/efs/CIGAR/data/05_results')
    imsave(debugs,fname='debug.png',fdir='/home/ubuntu/efs/CIGAR/data/04_vton_results')
    
    # save vton image and copy
    # glue all results
    vtons = torch.cat(results,dim=2)
    imsave(vtons,prefix='results',fdir='/home/ubuntu/efs/CIGAR/data/05_results')
    imsave(vtons,fname='results.png',fdir='/home/ubuntu/efs/CIGAR/data/04_vton_results')

    print('[Saved Image] Image saved!!')
    # save_image(vton,pth)
    # imsave(vton,prefix='result',fdir='./')
    # res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose[1])
    # imsave(res,fname='pose1.png',fdir='./')
    # plt.imsave('hi3.png',res) 
    # save_image(res,'hi.png')
    
    # res = plot_img(pimg=usr,gimgs=gars,gen_img=vton,pose=user_pose[1])
    return vtons
    
# """Code to Dress up specified garmnet on User image"""

# from dior.test_code.pose import *
# # from .datasets.deepfashion_datasets import *
# from torchvision.utils import save_image
# from torchvision import transforms
# from .datasets.human_parse_labels import get_label_map
# from .utils import pose_utils

# from PIL import Image, ImageOps
# import pandas as pd
# import copy

# class IngredientsLibrary():
#     def __init__(self,usr_dir,gar_dir,parse_dir,pose_csv,crop_size=(256,256)):
#         self.usr_dir = usr_dir
#         self.gar_dir = gar_dir
#         self.parse_dir = parse_dir
#         self.pose_csv = pose_csv
#         self.crop_size = crop_size
#         # transforms
#         self.resize = transforms.Resize(crop_size)
#         self.toTensor = transforms.ToTensor()
#         self.toPIL = transforms.ToPILImage()
#         self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       
#         self.annotation_file = pd.read_csv(pose_csv,sep=':').set_index('name')
    
#         self.aiyu2atr, self.atr2aiyu = get_label_map(8)

#     def _load_img(self, fn):
#         img = Image.open(fn).convert("RGB")
#         img = ImageOps.exif_transpose(img)
#         self.old_size = np.array(img).shape[:2]
#         img = self.resize(img)
#         img = self.toTensor(img)
#         img = self.normalize(img)
#         return img
    
#     def _load_mask(self, fn): 
#         mask = Image.open(fn + ".png")
#         mask = self.resize(mask)
#         mask = torch.from_numpy(np.array(mask))

#         texture_mask = copy.deepcopy(mask)
#         for atr in self.atr2aiyu:
#             aiyu = self.atr2aiyu[atr]
#             texture_mask[mask == atr] = aiyu
#         return texture_mask
    
#     def _load_kpt(self, name):
#         string = self.annotation_file.loc[name]
#         array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
#         pose  = pose_utils.cords_to_map(array, self.crop_size, self.old_size)
#         pose = np.transpose(pose,(2, 0, 1))
#         pose = torch.Tensor(pose)
#         return pose  
        
    
#     def load_item(self, key):
#         # import pdb;pdb.set_trace()
#         pth_img = os.path.join(self.usr_dir,key)
#         if not os.path.exists(pth_img):
#             pth_img = os.path.join(self.gar_dir,key)
#         img = self._load_img(pth_img)
#         kpt = self._load_kpt(key)     
#         parse = self._load_mask(os.path.join(self.parse_dir, key[:-4]))
#         return img.cuda(), kpt.cuda(), parse.cuda()

# def dress_up(ds,user,g_keys,g_cats,order=[5,1,3,2]):
    
#     PID=[0,4,6,7]
# #    import pdb;pdb.set_trace()
    
#     pimg, from_pose, parse = ds.load_item(user)
#     to_pose = from_pose
#     # load img,pose,parsemap for each entry

#     psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)
#     # encode base garments
#     gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
   
#     # swap base garment if any
#     gimgs = []
#     for gkey,gid in zip(g_keys,g_cats[1:]): # in gids:
#         gimg, pose, gparse = ds.load_item(gkey) #garment)load_img(gid, ds)
#         seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid)
#         # override garments
#         gsegs[gid] = seg
#         gimgs += [gimg * (gparse == gid)]

    
#     gsegs = [gsegs[i] for i in order]
#     gen_img = model.netG(to_pose[None], psegs, gsegs)

#     return pimg, gimgs, gen_img[0]

# def extract_vton(user,garments,gids,usr_dir,gar_dir,parse_dir,pose_csv):
#     """
#         user_pth: pth of user image
#         garments: list of paths to garment image
#     """
#     ds = IngredientsLibrary(usr_dir,gar_dir,parse_dir,pose_csv)

#     user = user.split('/')[-1]
#     g_pths = [file.split('/')[-1] for file in garments]
#     usr,gars,vton = dress_up(ds,user,g_pths,gids)
    
# #    res = plot_img(usr,gars,[],vton)
    
# #    plt.imsave('shi.png',res)
# #    save_image(res,'hi.png')
#     return vton
#     # import pdb;pdb.set_trace()