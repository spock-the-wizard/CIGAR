import argparse
import os
import json
import itertools
import sys 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.util import extract_subset


def main(args):
    with open(args.attribute2_idx_path, 'r') as f:
        attr2idx = json.load(f)

    top_k = args.top_k
    dress_top_attributes = dict()
    shirt_top_attributes = dict()
    toptee_top_attributes = dict()

    for category in ["dress","shirt", "toptee"]:
        for split in ["train", "val"]:
            file_name = f"asin2attr.{category}.{split}.json"
            file_path = os.path.join(args.fashioniq_path, file_name)
            with open(file_path, 'r') as f:
                fashion = json.load(f)
            for k,v in fashion.items():
                attr_list = list(itertools.chain(*v))
                for attr in attr_list:
                    if attr in attr2idx:
                        if category == "dress":
                            if attr not in dress_top_attributes:
                                dress_top_attributes[attr] = 1
                            else:
                                dress_top_attributes[attr] +=1
                        elif category == "shirt":
                            if attr not in shirt_top_attributes:
                                shirt_top_attributes[attr] = 1
                            else:
                                shirt_top_attributes[attr] +=1
                        elif category == "toptee":
                            if attr not in toptee_top_attributes:
                                toptee_top_attributes[attr] = 1
                            else:
                                toptee_top_attributes[attr] +=1
                        else:
                            pass
    
    dress_top_attr_list = []
    shirt_top_attr_list = []
    toptee_top_attr_list = []

    for k,v in dress_top_attributes.items():
        dress_top_attr_list.append((k,v))

    for k,v in shirt_top_attributes.items():
        shirt_top_attr_list.append((k,v))

    for k,v in toptee_top_attributes.items():
        toptee_top_attr_list.append((k,v))

    dress_top_attr_list.sort(key=lambda y:y[1], reverse=True)
    shirt_top_attr_list.sort(key=lambda y:y[1], reverse=True)
    toptee_top_attr_list.sort(key=lambda y:y[1], reverse=True)

    dress_top_attrs = []
    shirt_top_attrs = []
    toptee_top_attrs = []

    for attr_name, count in dress_top_attr_list:
        dress_top_attrs.append(attr_name)

    for attr_name, count in shirt_top_attr_list:
        shirt_top_attrs.append(attr_name)

    for attr_name, count in toptee_top_attr_list:
        toptee_top_attrs.append(attr_name)

    print(f"Dress: {dress_top_attrs[:top_k]}")
    print(f"Shirt: {shirt_top_attrs[:top_k]}")
    print(f"TopTee: {toptee_top_attrs[:top_k]}")


    # Now, we calculate the top attributes for lower-body garments
    lower_top_attributes =dict()
    subset = extract_subset("/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/list_category_img.txt", "/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/list_attr_img.txt")

    idx2attr = {v: k for k, v in attr2idx.items()}
    for elem in subset:
        file_name, attr_list = elem[0], elem[1]
        for idx, attr_v in enumerate(attr_list):
            if attr_v == 1:
                attr_name = idx2attr[idx]
                if attr_name not in lower_top_attributes:
                    lower_top_attributes[attr_name] = 1
                else:
                    lower_top_attributes[attr_name] +=1

    lower_top_attr_list = []

    for k,v in lower_top_attributes.items():
        lower_top_attr_list.append((k,v))

    lower_top_attr_list.sort(key=lambda y:y[1], reverse=True)
    lower_top_attrs = []

    for attr_name, count in lower_top_attr_list:
        lower_top_attrs.append(attr_name)

    print(f"Lower: {lower_top_attrs[:top_k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq_path", type=str, help="Path to the fashioniq dataset", default="/home/deokhk/coursework/fashion-iq/data")
    parser.add_argument("--attribute2_idx_path", type=str, help="Path to attribute2idx.json file", default="/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/attribute2idx.json")
    parser.add_argument("--top_k", type=int, help="Top k attributes to select", default=20)
    args = parser.parse_args()
    main(args)    