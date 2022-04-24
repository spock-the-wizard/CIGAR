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

        
score = dict()
hyperopt = dict()

def main(args):
    # init
    init_env()
        
    date_key = str(datetime.now().strftime("%Y%m%d%H%M%S"))[2:]

    # load model
    print(f'Load model: {args.expr_name}')
    root_path = os.path.join(args.test_root, 'repo', args.expr_name)
    with open(os.path.join(root_path, 'args.json'), 'r') as f:
        largs = json.load(f)
        largs = easydict.EasyDict(largs)
        pprint(largs)
        image_size = largs.image_size
        texts = torch.load(os.path.join(root_path, 'best_model.pth'))['texts']
    if largs.method == 'text-only':
        from src.model.text_only import TextOnlyModel
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
        from src.model.tirg import TIRG
        model = TIRG(
            args=largs,
            backbone=largs.backbone,
            texts=texts,
            text_method=largs.text_method,
            fdims=largs.fdims,
            fc_arch='B',
            init_with_glove=True,
            test_root = args.test_root
            #loss_type=largs.loss_type,
        )
    elif largs.method == 'match-tirg':
        from src.model.match import MatchTIRG
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
        from src.model.match import MatchTextOnly
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

    SPLIT = 'test'
    #targets = ['dress', 'toptee', 'shirt']
    target = args.category
    print(f">> SPLIT: {SPLIT} / TARGET: {target}")

    from src.dataset import FashionIQUserDataset
    index_dataset = FashionIQUserDataset(
        test_root = args.test_root,
        candidate = args.c_id,
        caption = args.caption,
        data_root=args.data_root,
        image_size=image_size,
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

    from src.dataset import FashionIQUserDataset
    query_dataset = FashionIQUserDataset(
        test_root = args.test_root,
        candidate = args.c_id,
        caption = args.caption,
        data_root=args.data_root,
        image_size=image_size,
        split=SPLIT,
        target=target
    )
    # query 데이터 만드는 작업 -> 1개밖에 필요 없다!
    query_ids = []
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
    print(query_feats.shape, index_feats.T.shape)
    # calculate cosine similarity
    print('calculating cosine similarity score...')
    y_score = np.dot(query_feats, index_feats.T)    # query_feats = (1,600 or 2048)  / index_feats = #. test images(600 or 2048, 3818)
    y_score = y_score[0]
    y_indices = np.argsort(-1 * y_score)

    print(y_score.shape, y_indices.shape)
    _score = []
    _r = []
    for j in range(min(TOP_K, len(y_score))):
        index = y_indices[j]
        print(index)
        _r.append([
            str(index_ids[index]),
            float(y_score[index])
        ])
    _score.append([args.c_id, _r])
    score[target] = _score
    print(score[target])
    # save score to file.
    print(f'Dump top-{TOP_K} rankings to .pkl file ...')
    os.makedirs(f'./output_score/{date_key}_{args.expr_name}', exist_ok=True)
    with open(os.path.join(f'./output_score/{date_key}_{args.expr_name}', f'hyperopt.{SPLIT}.pkl'), 'wb') as f:
        pickle.dump(hyperopt, f)
    with open(os.path.join(f'./output_score/{date_key}_{args.expr_name}', f'score.{SPLIT}.pkl'), 'wb') as f:
        pickle.dump(score, f)
    print('Done.')


if __name__ == "__main__":
    # args for region
    parser = argparse.ArgumentParser('Test')
    
    # Common options.
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed', type=int, default=int(time.time()), help='manual seed')
    parser.add_argument('--data_root', required=True, type=str, default='/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/data', help='data root directory path')
    parser.add_argument('--test_root', required=True, type=str, default = '/home/piai/chan/largescale_multimedia/project/FashionIQChallenge2020/ours/train')
    parser.add_argument('--expr_name', default='devel', type=str, help='experiment name')
    parser.add_argument('--c_id', required=True, type=str, help='id for candidate image')
    parser.add_argument('--category', required=True, type=str, choices=['dress', 'toptee', 'shirt', 'pants'])
    parser.add_argument('--caption', required=True, type=str, help='user feedback')
    
    ## parse and save args.
    args, _ = parser.parse_known_args()

    # main.
    main(args)
