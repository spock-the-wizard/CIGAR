import os,sys

GAR_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0,GAR_DIR)

# from dior.dress import *
from garment_transfer.parser.extract_parse_map import * #simple_extractor import get_garment_class,parse_and_classify
from garment_transfer.pose.extract_pose import *

from torchvision.utils import save_image
# from garment_transfer.dior.test_code.pose import *


DATA_DIR = os.path.join(os.path.dirname(GAR_DIR),'data')
# import pdb;pdb.set_trace()

# print(DATA_DIR)
USER_DIR = os.path.join(DATA_DIR,'01_user_image')
GAR_DIR = os.path.join(DATA_DIR,'02_image_retrieval')
POSE_DIR = './'#os.path.join(DATA_DIR,'05_pose')
VTON_DIR = os.path.join(DATA_DIR,'04_vton_results')
PARSE_DIR = os.path.join(DATA_DIR,'03_image_parsed')

def main(user_imgs, gar_img, cats, pose_path=os.path.join(POSE_DIR,'pose.csv'),parse_dir='./data/03_image_parsed/'):
    """
    Input: 
    - user_img: 
    - gar_img: garment image path.
    Output: TryOn Image

    """
    if not isinstance(user_imgs,list):
        user_imgs = [user_imgs]
    items = [*user_imgs,gar_img]

    print('='*70)
    print('[START] GARMENT TRANSFER')
    # torch.cuda.empty_cache()

    # extract pose
    print('='*50)
    # print('[STEP 1. POSE] Pose Extraction')
    extract_pose(items,pose_path)

    # torch.cuda.empty_cache()
    # import pdb;pdb.set_trace()
    # Save parse map and get garment category 
    print('='*50)
    print('[STEP 2. PARSE] Parse Map Extraction')
    extract_parse_map(items,parse_dir)
    
    # vton time
    print('='*50)
    print('[STEP 3. VTON] Generate Results')
    
    from garment_transfer.dior.extract_vton import extract_vton
    vton = extract_vton(user_imgs,[gar_img],cats,USER_DIR,GAR_DIR,PARSE_DIR,pose_csv=pose_path)
    
    # vton = dress_up(user_img,gar_pth,parse_pth)
    # store in ./data/04_vton_results
    # pth = os.path.join(VTON_DIR,'result.png')
    # save_image(vton,pth)
    # imsave(vton,prefix='result',fdir='./')

    # save result
    # print('Saving the image...')
    # imsave(gen_img,fname='test.png',fdir='../data/00_test/results')

    
if __name__=="__main__":
    
    dummy_user_pth = os.path.join(DATA_DIR,'test','user.png')#os.path.join(USER_DIR,'test.png') #'01_user_image/test.jpg'
    dummy_garment_pth = os.path.join(DATA_DIR,'test','shirt.png')#GAR_DIR,'A.png') #'02_image_retreival/A.png'
    main(dummy_user_pth,dummy_garment_pth,parse_dir=os.path.join(DATA_DIR,'test')) #PARSE_DIR) #None,None)
    

