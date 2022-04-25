import os
import numpy as np
import pandas as pd
import pickle
import argparse
import json
from tqdm import tqdm
from scipy.spatial import distance

def describe_threshold_distribution(augmented_lower_dataset):
    thersholds = []
    for elem in augmented_lower_dataset:
        thersholds.append(elem['similarity'])
    df = pd.DataFrame(thersholds)
    print(df.describe())

def filter_dataset(augmented_lower_dataset, threshold):
    filtered = filter(lambda elem: elem['similarity'] >= threshold, augmented_lower_dataset)
    return list(filtered)


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def main(args):
    fashioniq_vector_file_names =  [f"dress_{args.vector_type}.pickle", f"shirt_{args.vector_type}.pickle", f"toptee_{args.vector_type}.pickle"]
    fashioniq_imgname_to_vector = {}
    for f_name in fashioniq_vector_file_names:
        f_path = os.path.join(args.vector_dir, f_name)
        with open(f_path, "rb") as f:
            vectors = pickle.load(f)
        fashioniq_imgname_to_vector = {**fashioniq_imgname_to_vector, **vectors}

    lower_vector_file_name = f"lower_{args.vector_type}.pickle"
    vector_path = os.path.join(args.vector_dir, lower_vector_file_name)
    with open(vector_path, "rb") as f:
        lower_vectors = pickle.load(f)


    fashioniq_dir = args.fashioniq_dir
    file_list = ["cap.dress.train.json", "cap.dress.val.json", "cap.shirt.train.json", "cap.shirt.val.json", "cap.toptee.train.json", "cap.toptee.val.json"]
    captions = []
    for file_name in file_list:
        path = os.path.join(fashioniq_dir, file_name)
        with open(path ,'r') as f:
            cap = json.load(f)
        captions += cap

    lower_vector_to_name = {v.tostring(): k for k, v in lower_vectors.items()}

    lower_vector_list = []
    for k ,v in lower_vectors.items():
        lower_vector_list.append(v)
    # Returned generated data has a format like..
    # [{"target":, "candidate":, "captions:" [..], "similarity": []}]
    lower_augmented_caption_datas = []

    for caption in tqdm(captions):
        discard = False
        target = caption['target']
        candidate = caption['candidate']
        fashioniq_captions = caption['captions']

        try:
            target_vector = fashioniq_imgname_to_vector[target]
        except KeyError:
            discard = True
            
        try:
            candidate_vector = fashioniq_imgname_to_vector[candidate]
        except KeyError:
            discard = True

        if not discard:
            target_closest_vector = closest_node(target_vector, lower_vector_list)
            candidate_closest_vector = closest_node(candidate_vector, lower_vector_list)
            lower_target_imageid = lower_vector_to_name[target_closest_vector.tostring()]
            candidate_target_imageid = lower_vector_to_name[candidate_closest_vector.tostring()]

            vector_diff_fashioniq = target_vector - candidate_vector
            vector_diff_lower = target_closest_vector - candidate_closest_vector

            similarity = distance.cosine(vector_diff_fashioniq, vector_diff_lower)

            generated_data_pair = dict()
            generated_data_pair['target'] = lower_target_imageid
            generated_data_pair['candidate'] = candidate_target_imageid
            generated_data_pair['captions'] = fashioniq_captions
            generated_data_pair['similarity'] = similarity
            lower_augmented_caption_datas.append(generated_data_pair)
    
    print(f"Total generated dataset: {len(lower_augmented_caption_datas)}")
    describe_threshold_distribution(lower_augmented_caption_datas)

    filtered_dataset = filter_dataset(lower_augmented_caption_datas, args.similarity_threshold)
    print(f"Number of data that satisfies similarity > {args.similarity_threshold} : {len(filtered_dataset)}")
    
    save_file_name = f"augmented_{args.similarity_threshold}_{args.vector_type}.json"
    save_path = os.path.join(args.save_dir, save_file_name)
    with open(save_path, "w") as f:
        json.dump(filtered_dataset, f, indent=4, sort_keys=True)
    print(f"Saved filterd Augmented data to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", help = "Path to the directory containing the vectors generated from attributes", default ="/home/deokhk/coursework/CIGAR/data/image_retrieval")
    parser.add_argument("--vector_type", choices=["one-hot", "Bert", "Word2Vec"], help="Attribute vectors type", default="one-hot")
    parser.add_argument("--fashioniq_dir", help = "Path to the fasioniq caption dataset", default="/home/deokhk/coursework/fashion-iq/data/captions/")
    parser.add_argument("--save_dir", help="path to the directory to save the generated data", default="/home/deokhk/coursework/CIGAR/data/image_retrieval/generated_captions/")
    parser.add_argument("--similarity_threshold", default=0.0, type=float)
    args = parser.parse_args()
    main(args)