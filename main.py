import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import time
from image_retrieval.image_retrieval import ir_show_keywords, ir_find_match
from garment_transfer.garment_transfer import gt_generate
from utils.util import name_to_filepath
from image_retrieval.retrieval_model.ours.tools import load_model, load_gallery


USER_IMG_DIR = './data/01_user_image'
IMAGE_RETREIVAL_DIR = './data/02_image_retrieval'
IR_PARAMS = {'gpu_id': '0', 'manualSeed': int(time.time()), 
            'data_root': '/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/data',
            'test_root': '/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/ours/train',
            'expr_name': 'devel', 'img_size': 224}  # data_root, test_root 서버 경로로 바꿔야함!!

#c_id = 'B007IAPK1E'
#category = 'dress'
#caption = 'is a long black dress'


if __name__ == "__main__":

    print("="*50)
    user_img_pth = input('[Step 1] Enter user image name\n')
    # TODO: check validity of path
    # test path
    user_img_pth = './data/01_user_image/test.jpg'
    user_img = cv2.imread(user_img_pth)

    ir_model = load_model(gpu_id = IR_PARAMS.gpu_id,
                    manualSeed = IR_PARAMS.manualSeed,
                    test_root = IR_PARAMS.test_root,
                    expr_name = IR_PARAMS.expr_name)
   

    index_ids, index_feats = load_gallery(gpu_id = IR_PARAMS.gpu_id, 
                                    maualSeed = IR_PARAMS.manualSeed, 
                                    data_root = IR_PARAMS.data_root,
                                    test_root = IR_PARAMS.test_root,
                                    img_size = IR_PARAMS.img_size, 
                                    model = ir_model
                                    )

    garment = None
    feedback = None
    while True:
        # show ir_show_keyword()
        if garment is None:
            # show keywords and get user choice
            category, feedback = ir_show_keywords()

        # retrieval results - possibly with text attributes
        garment = ir_find_match(feedback, garment, category, ir_model, index_ids, index_feats, IR_PARAMS)

        # garment transfer, show results
        gt_generate(user_img,garment)

        response = input('[Step 2] Continue search?[Y/n]: ')
        if response != 'Y' and response != 'y':
            print('Ending search...')
            break
        else:
            feedback = input('[Step 3] Enter feedback: ')
            print('Received feedback: %s' % feedback)


    



