import argparse
import os
import json
import itertools



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

    print(f"Dress: {dress_top_attr_list[:top_k]}")
    print(f"Shirt: {shirt_top_attr_list[:top_k]}")
    print(f"TopTee: {toptee_top_attr_list[:top_k]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq_path", type=str, help="Path to the fashioniq dataset", default="/home/deokhk/coursework/fashion-iq/data")
    parser.add_argument("--attribute2_idx_path", type=str, help="Path to attribute2idx.json file", default="/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/attribute2idx.json")
    parser.add_argument("--top_k", type=int, help="Top k attributes to select", default=20)
    args = parser.parse_args()
    main(args)    