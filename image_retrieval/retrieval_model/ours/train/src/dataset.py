# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')
import random
random.seed(1234)
import operator
import numpy as np
import json
import operator
import pickle

from collections import defaultdict
from tqdm import tqdm
import urllib.request
from PIL import Image
from pprint import pprint

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.datasets as D
import torchvision.models as M
import torchvision.transforms.functional as TF

from src.transform import PaddedResize


class FashionIQDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 image_size=224,
                 split='val',
                 target='all'):
        self.data_root = data_root
        self.target = target
        self.image_size = image_size
        self.split = split
        self.transform = None
        self.all_targets = ['dress', 'toptee', 'shirt']

        self.reload()

    def __set_transform__(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.split in ['test', 'val']:
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __load_pil_image__(self, path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB', (224, 224))
            return img

    def __crop_image__(self, img, bbox):
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = y_min + int(h * bbox[3])
        crop_img = img.crop((x_min, y_min, x_max, y_max))
        return crop_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def reload(self):
        # load data
        self.__set_transform__()
        self.__load_data__()
        self.__print_status__()

    def __print_status__(self):
        print('=========================')
        print('Data Statistics:')
        print(f'{self.split} Data Size: {len(self.dataset)}')
        print('=========================')

    def get_loader(self, **kwargs):
        '''get_loader
        '''
        batch_size = kwargs.get('batch_size', 16)
        num_workers = kwargs.get('workers', 20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True

        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader

    def __load_data__(self):
        raise NotImplementedError()

    def __sample__(self, index):
        raise NotImplementedError()


class DeepFashionTrainDataset(FashionIQDataset):
    def __init__(self, deepfashion_datapath,
                 image_size,
                 split,
                 target,
                 caption_directory,
                 caption_file_name):
        self.caption_directory = caption_directory
        self.caption_file_name = caption_file_name
        super(DeepFashionTrainDataset, self).__init__(deepfashion_datapath, image_size, split, target)
        
    def __load_data__(self):
        with open('assets/sentence_embedding/embeddings.pkl', 'rb') as f:
            self.we = pickle.load(f)
        with open('assets/image_embedding/embeddings.pkl', 'rb') as f:
            self.ie = pickle.load(f)

        print('[Dataset] load caption annotations: {}'.format(self.caption_directory))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()

        if (self.target == 'all') or (self.target is None):
            targets = self.all_targets
        else:
            targets = [self.target]

        for t in targets:
            cap_file = self.caption_file_name

            print(f'[Dataset] load annotation file: {cap_file}')
            full_cap_path = os.path.join(self.caption_directory, cap_file)
            assert os.path.exists(full_cap_path)
            with open(full_cap_path, 'r') as f:
                data = json.load(f)
                for i, d in enumerate(tqdm(data)):
                    c_id = d['candidate']
                    t_id = d['target']

                    if not c_id in self.cls2idx:
                        self.cls2idx[c_id] = len(self.cls2idx)
                        self.idx2cls.append(c_id)
                    if not t_id in self.cls2idx:
                        self.cls2idx[t_id] = len(self.cls2idx)
                        self.idx2cls.append(t_id)

                    self.all_texts.extend(d['captions'])
                    text = [x.strip() for x in d['captions']]
                    random.shuffle(text)
                    text = '[CLS] ' + ' [SEP] '.join(text)
                    we_key = f'{self.split}_{t}_{c_id}_{i}'
                    c_img_path = os.path.join(self.data_root, c_id)
                    t_img_path = os.path.join(self.data_root, t_id)
                    if os.path.exists(c_img_path) and os.path.exists(t_img_path):
                        _data = {
                            'c_img_path': c_img_path,
                            'c_id': c_id,
                            't_img_path': t_img_path,
                            't_id': t_id,
                            'we_key': we_key,
                            'text': text
                        }

                        self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self, index):
        data = self.dataset[index]
        x_c = self.__load_pil_image__(data['c_img_path'])
        c_c = self.cls2idx[data['c_id']]
        x_t = self.__load_pil_image__(data['t_img_path'])
        c_t = self.cls2idx[data['t_id']]

        we_key = data['we_key']
        if we_key in self.we:
            we = torch.FloatTensor(self.we[data['we_key']])
        else:
            we = torch.zeros((600))

        t_id = data['t_id']
        if t_id in self.ie:
            ie = torch.FloatTensor(self.ie[data['t_id']])
        else:
            ie = torch.zeros((2048))

        if not self.transform is None:
            x_c = self.transform(x_c)
            x_t = self.transform(x_t)

        return (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, data['we_key'], data['text']),
            (ie)
        )

class FashionIQTrainValDataset(FashionIQDataset):
    def __init__(self, **kwargs):
        super(FashionIQTrainValDataset, self).__init__(**kwargs)

    def __load_data__(self):
        with open('assets/sentence_embedding/embeddings.pkl', 'rb') as f:
            self.we = pickle.load(f)
        with open('assets/image_embedding/embeddings.pkl', 'rb') as f:
            self.ie = pickle.load(f)

        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()

        if (self.target == 'all') or (self.target is None):
            targets = self.all_targets
        else:
            targets = [self.target]

        for t in targets:
            cap_file = f'captions/cap.{t}.{self.split}.json'

            print(f'[Dataset] load annotation file: {cap_file}')
            full_cap_path = os.path.join(self.data_root, cap_file)
            assert os.path.exists(full_cap_path)
            with open(full_cap_path, 'r') as f:
                data = json.load(f)
                for i, d in enumerate(tqdm(data)):
                    c_id = d['candidate']
                    t_id = d['target']

                    if not c_id in self.cls2idx:
                        self.cls2idx[c_id] = len(self.cls2idx)
                        self.idx2cls.append(c_id)
                    if not t_id in self.cls2idx:
                        self.cls2idx[t_id] = len(self.cls2idx)
                        self.idx2cls.append(t_id)

                    self.all_texts.extend(d['captions'])
                    text = [x.strip() for x in d['captions']]
                    random.shuffle(text)
                    text = '[CLS] ' + ' [SEP] '.join(text)
                    we_key = f'{self.split}_{t}_{c_id}_{i}'
                    c_img_path = os.path.join(self.data_root, f"images/{c_id}.jpg")
                    t_img_path = os.path.join(self.data_root, f"images/{t_id}.jpg")
                    if os.path.exists(c_img_path) and os.path.exists(t_img_path):
                        _data = {
                            'c_img_path': os.path.join(self.data_root, f"images/{c_id}.jpg"),
                            'c_id': c_id,
                            't_img_path': os.path.join(self.data_root, f"images/{t_id}.jpg"),
                            't_id': t_id,
                            'we_key': we_key,
                            'text': text
                        }

                        self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self, index):
        data = self.dataset[index]
        x_c = self.__load_pil_image__(data['c_img_path'])
        c_c = self.cls2idx[data['c_id']]
        x_t = self.__load_pil_image__(data['t_img_path'])
        c_t = self.cls2idx[data['t_id']]

        we_key = data['we_key']
        if we_key in self.we:
            we = torch.FloatTensor(self.we[data['we_key']])
        else:
            we = torch.zeros((600))

        t_id = data['t_id']
        if t_id in self.ie:
            ie = torch.FloatTensor(self.ie[data['t_id']])
        else:
            ie = torch.zeros((2048))

        if not self.transform is None:
            x_c = self.transform(x_c)
            x_t = self.transform(x_t)

        return (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, data['we_key'], data['text']),
            (ie)
        )


class FashionIQTestDataset(FashionIQDataset):
    def __init__(self, test_root, **kwargs):
        self.test_root = test_root
        super(FashionIQTestDataset, self).__init__(**kwargs)
    def __load_data__(self):
        with open(os.path.join(self.test_root, 'assets/sentence_embedding/embeddings.pkl'), 'rb') as f:
            self.we = pickle.load(f)
        split_file = f'image_splits/split.{self.target}.{self.split}.json'
        print(f'[Dataset] load split file: {split_file}')
        with open(os.path.join(self.data_root, split_file), 'r') as f:
            self.index_dataset = json.load(f)
        self.index_dataset = np.asarray(self.index_dataset)

        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.query_dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        cap_file = f'captions/cap.{self.target}.{self.split}.json'
        print(f'[Dataset] load annotation file: {cap_file}')
        with open(os.path.join(self.data_root, cap_file), 'r') as f:
            data = json.load(f)
            for i, d in enumerate(tqdm(data)):
                if not 'target' in d:
                    d['target'] = d['candidate']  # dummy

                c_id = d['candidate']
                t_id = d['target']

                if not c_id in self.cls2idx:
                    self.cls2idx[c_id] = len(self.cls2idx)
                    self.idx2cls.append(c_id)
                if not t_id in self.cls2idx:
                    self.cls2idx[t_id] = len(self.cls2idx)
                    self.idx2cls.append(t_id)

                self.all_texts.extend(d['captions'])
                text = [x.strip() for x in d['captions']]
                random.shuffle(text)
                text = '[CLS] ' + ' [SEP] '.join(text)

                we_key = f'{self.split}_{self.target}_{c_id}_{i}'
                c_img_path = os.path.join(self.data_root, f"images/{c_id}.jpg")
                t_img_path = os.path.join(self.data_root, f"images/{t_id}.jpg")
                if os.path.exists(c_img_path) and os.path.exists(t_img_path):
                    _data = {
                        'c_img_path': os.path.join(self.data_root, f"images/{c_id}.jpg"),
                        'c_id': c_id,
                        't_img_path': os.path.join(self.data_root, f"images/{t_id}.jpg"),
                        't_id': t_id,
                        'we_key': we_key,
                        'text': text
                    }

                    self.query_dataset.append(_data)
        self.query_dataset = np.asarray(self.query_dataset)

    def set_mode(self, mode):
        assert mode in ['query', 'index']
        self.mode = mode

    def __len__(self):
        if self.mode == 'query':
            return len(self.query_dataset)
        else:
            return len(self.index_dataset)

    def __print_status__(self):
        print('=========================')
        print('Data Statistics:')
        print(f'{self.split} Index Data Size: {len(self.index_dataset)}')
        print(f'{self.split} Query Data Size: {len(self.query_dataset)}')
        print('=========================')

    def __sample__(self, index):
        if self.mode == 'query':
            return self.__sample_query__(index)
        else:
            return self.__sample_index__(index)

    def __sample_query__(self, index):
        data = self.query_dataset[index]
        x_c = self.__load_pil_image__(data['c_img_path'])
        c_c = self.cls2idx[data['c_id']]
        x_t = self.__load_pil_image__(data['t_img_path'])
        c_t = self.cls2idx[data['t_id']]
        we = torch.FloatTensor(self.we[data['we_key']])

        if not self.transform is None:
            x_c = self.transform(x_c)
            x_t = self.transform(x_t)

        return (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, data['we_key'], data['text']),
        )

    def __sample_index__(self, index):
        image_id = self.index_dataset[index]
        x = self.__load_pil_image__(os.path.join(self.data_root, f'images/{image_id}.jpg'))

        if not self.transform is None:
            x = self.transform(x)

        return (x, image_id)


class FashionIQValIDDataset(FashionIQDataset):
    def __init__(self,
                 data_root,
                 target,
                 split='val',
                 **kwargs):
        super(FashionIQValIDDataset, self).__init__(data_root,
                                                    None,
                                                    split=split,
                                                    target=target,
                                                    **kwargs)

    def __load_data__(self):
        split_file = f'image_splits/split.{self.target}.{self.split}.json'
        # print(f'[Dataset] load split file: {split_file}')
        with open(os.path.join(self.data_root, split_file), 'r') as f:
            self.index_dataset = json.load(f)
        self.index_dataset = np.asarray(self.index_dataset)

        # print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.query_dataset = []
        self.cls2idx = dict()
        self.idx2cls = list()
        cap_file = f'captions/cap.{self.target}.{self.split}.json'
        # print(f'[Dataset] load annotation file: {cap_file}')
        with open(os.path.join(self.data_root, cap_file), 'r') as f:
            data = json.load(f)
            for i, d in enumerate(data):
                if not 'target' in d:
                    d['target'] = d['candidate']  # dummy

                c_id = d['candidate']
                t_id = d['target']

                if not c_id in self.cls2idx:
                    self.cls2idx[c_id] = len(self.cls2idx)
                    self.idx2cls.append(c_id)
                if not t_id in self.cls2idx:
                    self.cls2idx[t_id] = len(self.cls2idx)
                    self.idx2cls.append(t_id)

                we_key = f'{self.split}_{self.target}_{c_id}_{i}'
                _data = {
                    'c_id': c_id,
                    't_id': t_id,
                    'we_key': we_key,
                }

                self.query_dataset.append(_data)
        self.query_dataset = np.asarray(self.query_dataset)

    def __len__(self):
        return len(self.query_dataset)

    def __print_status__(self):
        return

    def __sample__(self, index):
        data = self.query_dataset[index]
        return (data['c_id'], data['t_id'], data['we_key'])


if __name__ == "__main__":
    pass



class FashionIQUserDataset(FashionIQDataset):
    def __init__(self, test_root, candidate, caption, **kwargs):
        self.test_root = test_root
        self.candidate = candidate
        self.caption = caption
        super(FashionIQUserDataset, self).__init__(**kwargs)

    #we에서 sentence embedding을 찾는게 아니라 모델로부터 직접 받아야함..
    def __load_data__(self):
        #with open(os.path.join(self.test_root, 'assets/sentence_embedding/embeddings.pkl'), 'rb') as f:
        #    self.we = pickle.load(f)

        #test split 안에 들어있는 file 이름들 불러오기
        split_file = f'image_splits/split.{self.target}.{self.split}.json'
        print(f'[Dataset] load split file: {split_file}')
        with open(os.path.join(self.data_root, split_file), 'r') as f:
            self.index_dataset = json.load(f)
        self.index_dataset = np.asarray(self.index_dataset)

        # 어차피 query image랑 caption은 1개 -> cls2idx/idx2cls 필요없음
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.query_dataset = []
        self.all_texts = []
        #self.cls2idx = dict()
        #self.idx2cls = list()
        c_id = self.candidate
        #t_id = self.candidate   #어차피 target은 안쓰임 place holder...
        #if not c_id in self.cls2idx:
        #    self.cls2idx[c_id] = len(self.cls2idx)
        #    self.idx2cls.append(c_id)
        #if not t_id in self.cls2idx:
        #    self.cls2idx[t_id] = len(self.cls2idx)
        #    self.idx2cls.append(t_id)

        # all_texts 는 vocab을 만드는 기준이 된다(이미 glove vectors가 있으므로 필요한 단어들만 가지고 vocab을 만든다)
        self.all_texts.extend(self.caption)
        text = [self.caption.strip()]
        #random.shuffle(text)
        text = '[CLS] ' + ' [SEP] '.join(text)

        # we(sentence embedding)를 더이상 쓰지 않으므로 key도 필요 없다
        # we_key = f'{self.split}_{self.target}_{c_id}'
        _data = {
                    'c_img_path': os.path.join(self.data_root, f"images/{c_id}.jpg"),
                    'c_id': c_id,
                    #'t_img_path': os.path.join(self.data_root, f"images/{t_id}.jpg"),
                    #'t_id': t_id,
                    #'we_key': we_key,
                    'text': text
                }
        self.query_dataset.append(_data)
        self.query_dataset = np.asarray(self.query_dataset)

    def set_mode(self, mode):
        assert mode in ['query', 'index']
        self.mode = mode

    def __len__(self):
        if self.mode == 'query':
            return len(self.query_dataset)
        else:
            return len(self.index_dataset)

    def __print_status__(self):
        print('=========================')
        print('Data Statistics:')
        print(f'{self.split} Index Data Size: {len(self.index_dataset)}')
        print(f'{self.split} Query Data Size: {len(self.query_dataset)}')
        print('=========================')

    def __sample__(self, index):
        if self.mode == 'query':
            return self.__sample_query__(index)
        else:
            return self.__sample_index__(index)

    def __sample_query__(self, index):
        data = self.query_dataset[index]
        x_c = self.__load_pil_image__(data['c_img_path'])
        c_c = 0                             # place holder
        we = data['text']                   # place holder
        w_key = 'user_feedback_text'       # place holder
        #c_c = self.cls2idx[data['c_id']]
        #x_t = self.__load_pil_image__(data['t_img_path'])
        #c_t = self.cls2idx[data['t_id']]
        #we = torch.FloatTensor(self.we[data['we_key']])

        if not self.transform is None:
            x_c = self.transform(x_c)
            #x_t = self.transform(x_t)
        
        return((x_c, c_c, data['c_id']),
            (we, w_key, data['text']))

    def __sample_index__(self, index):
        image_id = self.index_dataset[index]
        x = self.__load_pil_image__(os.path.join(self.data_root, f'images/{image_id}.jpg'))

        if not self.transform is None:
            x = self.transform(x)

        return (x, image_id)
