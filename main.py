import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from image_retrieval.image_retrieval import ir_show_keywords, ir_find_match
from garment_transfer.garment_transfer import gt_generate
from utils.util import name_to_filepath


USER_IMG_DIR = './data/01_user_image'
IMAGE_RETREIVAL_DIR = './data/02_image_retrieval'

if __name__ == "__main__":

    print("="*50)
    user_img_pth = input('[Step 1] Enter user image name\n')
    # TODO: check validity of path
    # test path
    user_img_pth = './data/01_user_image/test.jpg'
    user_img = cv2.imread(user_img_pth)

    garment = None
    feedback = None
    while True:
        # show ir_show_keyword()
        if garment is None:
            # show keywords and get user choice
            feedback = ir_show_keywords()

        # retrieval results - possibly with text attributes
        garment = ir_find_match(feedback, garment)
    
        # garment transfer, show results
        gt_generate(user_img,garment)

        response = input('[Step 2] Continue search?[Y/n]: ')
        if response != 'Y' and response != 'y':
            print('Ending search...')
            break
        else:
            feedback = input('[Step 3] Enter feedback: ')
            print('Received feedback: %s' % feedback)


    



