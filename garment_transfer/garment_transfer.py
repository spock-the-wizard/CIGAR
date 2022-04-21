import os,sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR)
# sys.path.append(os.path.dirname(SCRIPT_DIR))

# from dior.dress import *
from parser.extract_parse_map import * #simple_extractor import get_garment_class,parse_and_classify
from pose.extract_pose import *
from dior.extract_vton import *

from torchvision.utils import save_image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),'data')

# print(DATA_DIR)
USER_DIR = os.path.join(DATA_DIR,'01_user_image')
GAR_DIR = os.path.join(DATA_DIR,'02_image_retrieval')
POSE_DIR = os.path.join(DATA_DIR,'05_pose')
VTON_DIR = os.path.join(DATA_DIR,'04_vton_results')
PARSE_DIR = os.path.join(DATA_DIR,'03_image_parsed')

def main(user_img, gar_img, pose_path=os.path.join(POSE_DIR,'pose.csv'),parse_dir='./data/03_image_parsed/'):
    """
    Input: 
    - user_img: 
    - gar_img: garment image path.
    Output: TryOn Image

    """
    items = [user_img,gar_img]

    print('='*70)
    print('[START] GARMENT TRANSFER')

    # extract pose
    print('='*50)
    print('[STEP 1. POSE] Pose Extraction')
    extract_pose(items,pose_path)

    # Save parse map and get garment category 
    print('='*50)
    print('[STEP 2. PARSE] Parse Map Extraction')
    cats = extract_parse_map(items,parse_dir)

    # vton time
    print('='*50)
    print('[STEP 3. VTON] Generate Results')
    vton = extract_vton(user_img,[gar_img],cats,USER_DIR,GAR_DIR,PARSE_DIR,pose_csv=pose_path)
    
    # vton = dress_up(user_img,gar_pth,parse_pth)
    # store in ./data/04_vton_results
    pth = os.path.join(VTON_DIR,'result.png')
    save_image(vton,pth)

    # save result
    print('Saving the image...')
    # imsave(gen_img,fname='test.png',fdir='../data/00_test/results')

    
if __name__=="__main__":
    
    dummy_user_pth = os.path.join(USER_DIR,'test.png') #'01_user_image/test.jpg'
    dummy_garment_pth = os.path.join(GAR_DIR,'A.png') #'02_image_retreival/A.png'
    main(dummy_user_pth,dummy_garment_pth,parse_dir=PARSE_DIR) #None,None)
    

