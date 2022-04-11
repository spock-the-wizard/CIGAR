import os
import json
from tqdm import tqdm

def name_to_filepath(dir,name):
    fnames = []
    for fname in os.listdir(dir):
        if fname.split('.')[-1]==name:
            fnames.append(os.path.join(dir,fname))

    return fnames

def create_attribute2idx(attribute_file_path="/home/deokhk/coursework/Category_and_Attribute/Anno_coarse/list_attr_cloth.txt"):
    """
    attribute_file_path : deepfashion attribute_list.txt
    """
    with open(attribute_file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    num_attr = lines[1]
    attribute2idx = dict()
    for idx, line in enumerate(lines[2:]):
        attr = line[:-2].strip()
        attr_end = attr[-1]
        if attr_end.endswith("1") or attr_end.endswith("2") or attr_end.endswith("3") or attr_end.endswith("4") or attr_end.endswith("5"):
            attr = attr[:-1].strip()
        attribute2idx[attr] = idx
    
    path = os.path.join(os.path.dirname(attribute_file_path), 'attribute2idx.json')
    with open(path, 'w') as json_file:
        json.dump(attribute2idx, json_file)

def extr_data_txt(path):
    """
    Load data from text file.
    each entry is a dictionary whose key is a imagepath and have a corresponding value.
    """
    print("Reading data txt..")
    with open(path, "r") as f:
        data = []
        for itr, line in tqdm(enumerate(f)):
            # Because we got annotation in the first two lines
            if itr >= 2:
                image_path, value = line.split()
                data.append([image_path, value])
    return data

def extr_data_txt_attr_img(path):
    """
    Load data from text file.
    suitable for list_attr_img.txt
    """
    print("Reading attr txt..")
    with open(path, "r") as f:
        data = []
        for itr, line in tqdm(enumerate(f)):
            # Because we got annotation in the first two lines
            if itr >= 2:
                lines = line.split()
                image_path = lines[0]
                attrs = [int(v) for v in lines[1:]]
                data.append([image_path, attrs])
    return data

def extract_subset(list_category_image_path, list_attr_image_path):
    category_lower_body_range = range(21, 37)
    data = extr_data_txt(list_category_image_path)
    subset = set()
    print("Now generating subset..")
    for e in tqdm(data):
        if int(e[1]) in category_lower_body_range:
            subset.add(e[0])

    print("Now making imagepath-attribute pair belongs to the subset")
    data = extr_data_txt_attr_img(list_attr_image_path)
    extracted_lower_attrs = []
    for e in tqdm(data):
        if e[0] in subset:
            extracted_lower_attrs.append(e)
    
    return extracted_lower_attrs

