import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import time
from image_retrieval.image_retrieval import ir_show_keywords, ir_find_match
from utils.util import name_to_filepath
from image_retrieval.retrieval_model.ours.tools.load_model import load_model
from image_retrieval.retrieval_model.ours.tools.load_gallery import load_gallery


USER_IMG_DIR = './data/01_user_image'
IMAGE_RETREIVAL_DIR = './data/02_image_retrieval'
IR_PARAMS = {'gpu_id': '0', 
            'manualSeed': int(time.time()), 
            'data_root': '/home/piai/chan/largescale_multimedia/project/CIGAR/image_retrieval/retrieval_model/data',
            'test_root': '/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/ours/train',
            'expr_name': 'devel', 
            'image_size': 224}  # data_root, test_root 서버 경로로 바꿔야함!!

#c_id = 'B007IAPK1E'
#category = 'dress'
#caption = 'is a long black dress'


if __name__ == "__main__":

    garment = None
    feedback = None
    print("="*50)
    print(IR_PARAMS)
    ir_model = load_model(IR_PARAMS)
    category, feedback = ir_show_keywords()
    IR_PARAMS['category'] = category
    index_ids, index_feats = load_gallery(ir_model, IR_PARAMS)

    while True:
        # retrieval results - possibly with text attributes
        garment = ir_find_match(feedback, garment, ir_model, index_ids, index_feats, IR_PARAMS)
        feedback = 'I want black one with longer sleeves'
        # garment transfer, show results
        print(garment)

    



