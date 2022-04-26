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

from utils.util import glue_images

USER_IMG_DIR = './data/01_user_image'
IMAGE_RETREIVAL_DIR = './data/02_image_retrieval'
IR_PARAMS = {'gpu_id': '0', 
            'manualSeed': int(time.time()), 
            'data_root': '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/data',
            'test_root': '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/ours/train',
            'deepfashion_root': '/home/ubuntu/efs',
            'deepfashion_caption_path': '/home/ubuntu/efs/CIGAR/data/image_retrieval/generated_captions/augmented_0.5_bert.json',
            'expr_name': 'devel', 
            'deepfashion_expr_name': 'bert_0.5',
            'image_size': 224}  # data_root, test_root 서버 경로로 바꿔야함!!

TEST_DRESS = ['B00A3Q8G8Y','B00599DYKA','B0035WTCA4']
if __name__ == "__main__":

    # glue_images()
    print("="*50)
    # user_img_pth = input('[Step 1] Enter user image name\n')
    # test path
    # user_img = './data/01_user_image/user.jpg'
    user_imgs = ['./data/01_user_image/user%d.jpg'%d for d in range(1)]
    # if not os.path.exists(user_img):
    #     print('File not exists!')

    garment = None
    feedback = None
    GAR_RAW_DIR = '/home/ubuntu/efs/CIGAR/image_retrieval/retrieval_model/data/images/'
    GAR_DIR = '/home/ubuntu/efs/CIGAR/data/02_image_retrieval/'1

    # upload all images
    category, feedback = ir_show_keywords()
    IR_PARAMS['category'] = category
    ir_model = load_model(IR_PARAMS)

    index_ids, index_feats = load_gallery(ir_model, IR_PARAMS)
    cats = []
    while True:
        # GT DEBUG TODO: COMMENT OUT
        # garment_id = TEST_DRESS[0]
        # category = 'dress'

        cats.append(category)
        garment_id = ir_find_match(feedback, garment, ir_model, index_ids, index_feats, IR_PARAMS)
        

        # garment transfer, show results
        garmentscr = GAR_RAW_DIR + garment_id + '.jpg'
        if not os.path.exists(garmentscr):
            print("Garment Image not exists")
        garment = os.path.join(GAR_DIR, garment_id+'.jpg')
        shutil.copyfile(garmentscr, garment)

        # garment = '/home/ubuntu/efs/CIGAR/data/02_image_retrieval/B00A16E1X0.jpg'
        # garment transfer
        gt_generate(user_imgs,garment,cats)

        response = input('[Step 2] Continue search ?[Y/n]: ')
        if response != 'Y' and response != 'y':
            print('Ending search...')
            break
        else:
            category = input('[Step 3] Search different category?[Y/n]: ')
            if category != 'Y' and category != 'y':
                feedback = input('[Step 4] Enter feedback: ')
                print('Received feedback: %s' % feedback)

            else:
                category, feedback = ir_show_keywords()
                IR_PARAMS['category'] = category
                index_ids, index_feats = load_gallery(ir_model, IR_PARAMS)
                cats = []


    



