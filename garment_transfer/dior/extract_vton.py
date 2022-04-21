"""Code to Dress up specified garmnet on User image"""

from dior.test_code.pose import *
from .datasets.deepfashion_datasets import *
from torchvision.utils import save_image
class IngredientsLibrary():
    def __init__(self,usr_dir,gar_dir,parse_dir,pose_csv,crop_size=(256,256)):
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
        self.old_size = np.array(img).shape[:2]
        img = self.resize(img)
        img = self.toTensor(img)
        img = self.normalize(img)
        
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
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose  = pose_utils.cords_to_map(array, self.crop_size, self.old_size)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose  
        
    
    def load_item(self, key):
        # import pdb;pdb.set_trace()
        pth_img = os.path.join(self.usr_dir,key)
        if not os.path.exists(pth_img):
            pth_img = os.path.join(self.gar_dir,key)
        img = self._load_img(pth_img)
        kpt = self._load_kpt(key)     
        parse = self._load_mask(os.path.join(self.parse_dir, key[:-4]))
        return img.cuda(), kpt.cuda(), parse.cuda()

def dress_up(ds,user,g_keys,g_cats,order=[5,1,3,2]):
    
    PID=[0,4,6,7]
    
    pimg, from_pose, parse = ds.load_item(user)
    to_pose = from_pose
    # load img,pose,parsemap for each entry

    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)
    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
   
    # swap base garment if any
    gimgs = []
    for gkey,gid in zip(g_keys,g_cats): # in gids:
        gimg, pose, gparse = ds.load_item(gkey) #garment)load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid)
        # override garments
        gsegs[gid] = seg
        gimgs += [gimg * (gparse == gid)]

    
    gsegs = [gsegs[i] for i in order]
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, gen_img[0]

def extract_vton(user,garments,gids,usr_dir,gar_dir,parse_dir,pose_csv):
    """
        user_pth: pth of user image
        garments: list of paths to garment image
    """
    ds = IngredientsLibrary(usr_dir,gar_dir,parse_dir,pose_csv)

    user = user.split('/')[-1]
    g_pths = [file.split('/')[-1] for file in garments]
    res = dress_up(ds,user,g_pths,gids)
    
    
    import pdb;pdb.set_trace()