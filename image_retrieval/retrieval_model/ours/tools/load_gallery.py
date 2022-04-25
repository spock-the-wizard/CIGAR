'''
get_score.py
'''
import os
import sys
sys.path.append('../train')
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
def init_env():
    # load argsuration.
    state = {k: v for k, v in args._get_kwargs()}
    pprint(state)

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.

def load_gallery(args):
    SPLIT = 'test'
    #targets = ['dress', 'toptee', 'shirt']
    target = args.category
    print(f">> SPLIT: {SPLIT} / TARGET: {target}")
    model = args.model
    from src.dataset import FashionIQUserDataset
    index_dataset = FashionIQUserDataset(
        test_root = args.test_root,
        candidate = args.c_id,
        caption = args.caption,
        data_root=args.data_root,
        image_size=args.image_size,
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
    

if __name__ == "__main__":
    # args for region
    parser = argparse.ArgumentParser('Test')
    
    # Common options.
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed', type=int, default=int(time.time()), help='manual seed')
    parser.add_argument('--data_root', required=True, type=str, default='/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/data', help='data root directory path')
    parser.add_argument('--test_root', required=True, type=str, default = '/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/ours/train')
    parser.add_argument('--img_size', type=int, help='image size for the trained model')
    parser.add_argument('--model', type=object, help='model instance')
    #parser.add_argument('--expr_name', default='devel', type=str, help='experiment name')
    
    ## parse and save args.
    args, _ = parser.parse_known_args()

    # main.
    load_gallery(args)
