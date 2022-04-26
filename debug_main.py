import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import time
import shutil
from image_retrieval.image_retrieval import ir_show_keywords, ir_find_match
from garment_transfer.garment_transfer import main as gt_generate
# from utils.util import name_to_filepath
from image_retrieval.retrieval_model.ours.tools.load_model import load_model
from image_retrieval.retrieval_model.ours.tools.load_gallery import load_gallery


USER_IMG_DIR = './data/01_user_image'
IMAGE_RETREIVAL_DIR = './data/02_image_retrieval'
IR_PARAMS = {'gpu_id': '0', 
            'manualSeed': int(time.time()), 
            'data_root': '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/data',
            'test_root': '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/ours/train',
            'expr_name': 'devel', 
            'image_size': 224}  # data_root, test_root 서버 경로로 바꿔야함!!

if __name__ == "__main__":

    print("="*50)
    user_img_pth = input('[Step 1] Enter user image name\n')
    # TODO: check validity of path
    # test path
    user_img = './data/01_user_image/user.png'
    if not os.path.exists(user_img):
        print('File not exists!')

    garment = None
    feedback = None
    GAR_RAW_DIR = '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/data/images/'
    GAR_DIR = '/home/ubuntu/efs/CIGAR/data/02_image_retrieval/'
    '''
    ir_model = load_model(IR_PARAMS)
    category, feedback = ir_show_keywords()
    IR_PARAMS['category'] = category
    import pdb;pdb.set_trace()
    index_ids, index_feats = load_gallery(ir_model, IR_PARAMS)
    # save all options
    import pdb;pdb.set_trace()
    '''
    cats = []
    while True:
        category = 'dress'
        cats.append(category)
        '''
        garment_id = ir_find_match(feedback, garment, ir_model, index_ids, index_feats, IR_PARAMS)
        
        # garment transfer, show results
        print(user_img, garment)
        garmentscr = GAR_RAW_DIR + garment_id + '.jpg'
        if not os.path.exists(garmentscr):
            print("Garment Image not exists")
        garment = os.path.join(GAR_DIR, garment_id+'.jpg')
        shutil.copyfile(garmentscr, garment)
        '''
        garment = '/home/ubuntu/efs/CIGAR/data/02_image_retrieval/B00A16E1X0.jpg'
        # garment transfer
        gt_generate(user_img,garment,cats)

        response = input('[Step 2] Continue search?[Y/n]: ')
        if response != 'Y' and response != 'y':
            print('Ending search...')
            break
        else:
            feedback = input('[Step 3] Enter feedback: ')
            print('Received feedback: %s' % feedback)


    



