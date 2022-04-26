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

score = dict()
hyperopt = dict()

def load_model(args):
    # init
    init_env(args)
    # load model
    print(f'Load model: {args["expr_name"]}')
    if args['category'] == 'bottom':
        root_path = os.path.join(args['test_root'], 'repo', args['deepfashion_expr_name'])
    else:
        root_path = os.path.join(args['test_root'], 'repo', args['expr_name'])
    with open(os.path.join(root_path, 'args.json'), 'r') as f:
        largs = json.load(f)
        largs = easydict.EasyDict(largs)
        pprint(largs)
        texts = torch.load(os.path.join(root_path, 'best_model.pth'))['texts']
    if largs.method == 'text-only':
        from train.src.model.text_only import TextOnlyModel
        model = TextOnlyModel(args=largs,
                              backbone=largs.backbone,
                              texts=texts,
                              text_method=largs.text_method,
                              fdims=largs.fdims,
                              fc_arch='A',
                              init_with_glove=False,
                              #loss_type=largs.loss_type
                              )
    elif largs.method == 'tirg':
        from train.src.model.tirg import TIRG
        model = TIRG(
            args=largs,
            backbone=largs.backbone,
            texts=texts,
            text_method=largs.text_method,
            fdims=largs.fdims,
            fc_arch='B',
            init_with_glove=True,
            test_root = args['test_root']
            #loss_type=largs.loss_type,
        )
    elif largs.method == 'match-tirg':
        from train.src.model.match import MatchTIRG
        model = MatchTIRG(
            args=largs,
            backbone=largs.backbone,
            texts=texts,
            text_method=largs.text_method,
            fdims=largs.fdims,
            fc_arch='B',
            init_with_glove=True,
            #loss_type=largs.loss_type,
        )
    elif largs.method == 'match-text-only':
        from train.src.model.match import MatchTextOnly
        model = TextOnlyModel(backbone=largs.backbone,
                              texts=texts,
                              text_method=largs.text_method,
                              fdims=largs.fdims,
                              fc_arch='A',
                              init_with_glove=False,
                              #loss_type=largs.loss_type
                              )
            
    model.load(os.path.join(root_path, 'best_model.pth'))
    model = model.cuda()
    model.eval()
    print(model)
    return model
