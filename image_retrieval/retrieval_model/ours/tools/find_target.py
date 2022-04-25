'''
get_score.py
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import time
import random
import json
import argparse
import easydict

from pprint import pprint
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from datetime import datetime

TOP_K = 10

# init
def init_env(args):
    # load argsuration.
    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    manualSeed = args['manualSeed']
    # Random seed
    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.

def find_target(model, args, index_ids, index_feats, c_id, caption):
    # init
    init_env(args)
    
    SPLIT = 'test'
    #targets = ['dress', 'toptee', 'shirt']
    target = args['category']
    print(f">> SPLIT: {SPLIT} / TARGET: {target}")

    from train.src.dataset import FashionIQUserDataset
    query_dataset = FashionIQUserDataset(
        test_root = args['test_root'],
        candidate = c_id,
        caption = caption,
        data_root=args['data_root'],
        image_size=args['image_size'],
        split=SPLIT,
        target=target
    )
    # query 데이터 만드는 작업 -> 1개밖에 필요 없다!
    query_feats = []
    print('Extract Query Features...')
    query_loader = query_dataset.get_loader(batch_size=1)
    query_loader.dataset.set_mode('query')
    for bidx, input in enumerate(tqdm(query_loader, desc='Query')):
        """
        input[0] = (x_c, c_c, data['c_id'])
        input[1] = (we, w_key, text)    
        tirg -> get_manipulated_image_feature(x)에서
        x[0] = (x_c, c_c, data['c_id'])
        x[1] = (we, w_key, text) 이런 모양을 기대중..
        """
        print('img size: ', input[0][0].shape) # [1, 3, 224, 224]
       
        input[0][0] = Variable(input[0][0].cuda())  # candidate 이미지
        with torch.no_grad():
            input[1][0] = model.extract_text_feature(input[1][2])   # we <- candidate 이미지에 대한 sentence embedding이 들어가야함
        input[1][0] = Variable(input[1][0].cuda())
        print('caption size: ', input[1][0].shape)
        # textencoder에 sentence넣어서 output받기
        data = (input[0], input[1])
        #print(data)
        with torch.no_grad():
            output = model.get_manipulated_image_feature(data) 

        for i in range(output.size(0)):
            # query
            #_qid = input[2][1][i]
            _feat = output[i].squeeze().cpu().numpy()
            query_feats.append(_feat)
            #query_ids.append(_qid)
        
    query_feats = np.asarray(query_feats)
    # calculate cosine similarity
    print('calculating cosine similarity score...')
    y_score = np.dot(query_feats, index_feats.T)    # query_feats = (1,600 or 2048)  / index_feats = #. test images(600 or 2048, 3818)
    y_score = y_score[0]
    y_indices = np.argsort(-1 * y_score)

    print(y_score.shape, y_indices.shape)
    score = []
    for j in range(min(TOP_K, len(y_score))):
        index = y_indices[j]
        print(index)
        score.append(
            str(index_ids[index])
        )
    return score
    # save score to file.
