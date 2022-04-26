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

def load_gallery(model, args):
    SPLIT = 'test'
    #targets = ['dress', 'toptee', 'shirt']
    target = args['category']
    print(f">> SPLIT: {SPLIT} / TARGET: {target}")

    from train.src.dataset import FashionIQTestDataset, DeepfashionTestDataset
    if args.category != 'bottom':
        index_dataset = FashionIQTestDataset(
            test_root = args['test_root'],
            data_root=args['data_root'],
            image_size=args['image_size'],
            split=SPLIT,
            target=target
        )
    else:
        index_dataset = DeepfashionTestDataset(
            test_root = args['test_root'],
            data_root=args['data_root'],
            image_size=args['image_size'],
            split=SPLIT,
            target=target
        )

    index_loader = index_dataset.get_loader(batch_size=16)

    index_ids = []
    index_feats = []
    print('Extract Index Features...')
    index_loader.dataset.set_mode('index')

    # gallery 만드는 작업
    for bidx, input in enumerate(tqdm(index_loader, desc='Index')):
        input[0] = Variable(input[0].cuda())      # input = (x, image_id)
        data = input[0]
        image_id = input[1]

        with torch.no_grad():
            output = model.get_original_image_feature(data)

        for i in range(output.size(0)):
            _iid = image_id[i]
            _feat = output[i].squeeze().cpu().numpy()
            index_feats.append(_feat)
            index_ids.append(_iid)

    index_feats = np.asarray(index_feats)
    return (index_ids, index_feats)