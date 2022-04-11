import numpy as np
import json
import argparse
import itertools
import os
import pickle
import sys

sys.path.append("/home/deokhk/coursework/CIGAR/")
from tqdm import tqdm
from utils.util import extract_subset

"""
This file saves pickle file with type dict(numpy).
Key in dict corresponds to imageid and value is a attribute vector.
Here, we generate three types of attribute vectors.
[1] Sum of one-hot vector [1000d] 
[2] Bert embedding given a list of attributes [512d]
[3] Sum of word vectors [256d]
"""

def convert_to_one_hot(dress_to_attr, shirt_to_attr, toptee_to_attr, attr2idx, deepfashion_img_category_path, deepfashion_img_attr_path):
    dress_to_vector = dict()
    shirt_to_vector = dict()
    toptee_to_vector = dict()
    lower_body_to_vector = dict()

    print("Now converting dress attributes to one-hot vector..")
    hitcount = 0

    for k, attr_list in tqdm(dress_to_attr.items()):
        zero_v = np.zeros(1000) # We have 1000 attributes
        for attr in attr_list:
            if attr in attr2idx:
                idx = attr2idx[attr]
                zero_v[idx] = 1
                hitcount+=1
        dress_to_vector[k] = zero_v

    print(f"Average {hitcount/len(dress_to_attr)} attributes selected.")


    print("Now converting shirt attributes to one-hot vector..")
    hitcount = 0

    for k, attr_list in tqdm(shirt_to_attr.items()):
        zero_v = np.zeros(1000) # We have 1000 attributes
        for attr in attr_list:
            if attr in attr2idx:
                idx = attr2idx[attr]
                zero_v[idx] = 1
                hitcount +=1
        shirt_to_vector[k] = zero_v

    print(f"Average {hitcount/len(shirt_to_attr)} attributes selected.")

    print("Now converting toptee attributes to one-hot vector..")
    hitcount = 0

    for k, attr_list in tqdm(toptee_to_attr.items()):
        zero_v = np.zeros(1000) # We have 1000 attributes
        for attr in attr_list:
            if attr in attr2idx:
                idx = attr2idx[attr]
                zero_v[idx] = 1
                hitcount +=1
        toptee_to_vector[k] = zero_v
    print(f"Average {hitcount/len(toptee_to_attr)} attributes selected.")

    subset = extract_subset(deepfashion_img_category_path, 
                            deepfashion_img_attr_path)

    print("Now converting lower-clothes attributes to one-hot vector..")
    hitcount = 0

    for elem in tqdm(subset):
        file_name, attr_list = elem[0], elem[1]
        zero_v = np.zeros(1000) # We have 1000 attributes
        for idx, attr_v in enumerate(attr_list):
            if attr_v == 1:
                zero_v[idx] = 1
                hitcount +=1
        lower_body_to_vector[file_name] = zero_v
    print(f"Average {hitcount/len(subset)} attributes selected.")


    return dress_to_vector, shirt_to_vector, toptee_to_vector, lower_body_to_vector
    


def convert_to_bert(dress_to_attr, shirt_to_attr, toptee_to_attr):
    raise NotImplementedError

def convert_to_wordvector(dress_to_attr, shirt_to_attr, toptee_to_attr):
    raise NotImplementedError

def save_to_vector_to_pickle(dress_vectors, shirt_vectors, toptee_vectors, lower_body_to_vector, mode, data_path):
    dress_filename = f"dress_{mode}.pickle"
    shirt_filename = f"shirt_{mode}.pickle"
    toptee_filename = f"toptee_{mode}.pickle"
    lower_filename = f"lower_{mode}.pickle"

    dress_path = os.path.join(data_path, dress_filename)
    shirt_path = os.path.join(data_path, shirt_filename)
    toptee_path = os.path.join(data_path, toptee_filename)
    lower_path = os.path.join(data_path, lower_filename)

    path_list = [dress_path, shirt_path, toptee_path, lower_path]
    vectors = [dress_vectors, shirt_vectors, toptee_vectors, lower_body_to_vector]
    for path, vector in zip(path_list, vectors):
        with open(path, 'wb') as f:
            pickle.dump(vector, f)
        print(f"Saved {path}.")

def main(args):
    with open(args.attribute2_idx_path, 'r') as f:
        attr2idx = json.load(f)
    dress_to_attr = dict()
    shirt_to_attr = dict()
    toptee_to_attr = dict()
    for category in ["dress","shirt", "toptee"]:
        for split in ["train", "val"]:
            file_name = f"asin2attr.{category}.{split}.json"
            file_path = os.path.join(args.fashioniq_path, file_name)
            with open(file_path, 'r') as f:
                fashion = json.load(f)            
            for k,v in fashion.items():
                attr_list = list(itertools.chain(*v))
                if category == "dress":
                    dress_to_attr[k] = attr_list
                elif category == "shirt":
                    shirt_to_attr[k] = attr_list
                else:
                    toptee_to_attr[k] = attr_list
    
    print(f"=== Converting fashioniq attributes to {args.mode} embeddings === ")
    if args.mode == "one-hot":
        dress_vectors, shirt_vectors, toptee_vectors, lower_body_to_vector = convert_to_one_hot(dress_to_attr, shirt_to_attr, toptee_to_attr, attr2idx,
        "/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/list_category_img.txt", "/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/list_attr_img.txt")
        save_to_vector_to_pickle(dress_vectors, shirt_vectors, toptee_vectors, lower_body_to_vector, mode=args.mode, data_path="/home/deokhk/coursework/CIGAR/data/image_retrieval/")

    elif args.mode == "bert":
        pass
        #dress_vectors, shirt_vectors, toptee_vectors = convert_to_bert(dress_to_attr, shirt_to_attr, toptee_to_attr)
    elif args.mode == "word-vector":
        pass
        #dress_vectors, shirt_vectors, toptee_vectors = convert_to_wordvector(dress_to_attr, shirt_to_attr, toptee_to_attr)
    else:
        raise ValueError("Invalid attribute vector type!")

    # Save converted fashionIQ vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq_path", type=str, help="Path to the fashioniq dataset", default="/home/deokhk/coursework/fashion-iq/data")
    parser.add_argument("--deepfashion_path", type=str, help="Path to the deepfashion dataset", default="")
    parser.add_argument("--attribute2_idx_path", type=str, help="Path to attribute2idx.json file", default="/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/attribute2idx.json")
    parser.add_argument("--mode", choices=["one-hot", "bert", "word-vector"], default="one-hot")
    args = parser.parse_args()
    main(args)