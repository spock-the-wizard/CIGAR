import numpy as np
import os
import pickle
from torch import imag, manual_seed
from image_retrieval.retrieval_model.ours.tools.find_target import find_target, find_target_deepfashion
from utils.util import glue_images
attribute_dict = {'dress': ['wash', 'clean', 'sleeve', 'sleeveless', 'print', 'fit', 'cotton', 'maxi', 'printed', 'shoulder', 'lace', 'zipper', 'hem', 'neckline', 'please', 'strapless', 'chiffon', 'stretch', 'party', 'v-neck'],
'shirt': ['cotton', 'wash', 'shirt', 'sleeve', 'fit', 'printed', 'logo', 'long sleeve', 'print', 'collar', 'pocket', 'button', 'soft', 'graphic', 'woven', 'classic', 'polo', 'crew', 'hem', 'stripe'],
'toptee': ['wash', 'cotton', 'sleeve', 'shirt', 'print', 'fit', 'printed', 'long sleeve', 'scoop', 'soft', 'button', 'clean', 'hem', 'v-neck', 'knit', 'neckline', 'sleeveless', 'lace', 'shoulder', 'tunic'],
'bottom': ['denim', 'print', 'skinny', 'floral', 'wash', 'drawstring', 'mini', 'pencil', 'distressed', 'lace', 'leather', 'pleated', 'knit', 'classic', 'skater', 'maxi', 'faux', 'striped', 'acid', 'cotton']}

def ir_show_keywords():
    """
    Input: X
    Output: 
        selected_keywords / list(str)

    In this function, we should show 
    """
    categories = ['dress', 'shirt', 'toptee', 'bottom']
    for idx, keyword in enumerate(categories):
        print(f"{idx}. {keyword}")
    category_idx = int(input("Select one category: ")) # We assume user input has format like "1,2,3,4"
    selected_category = categories[category_idx]
    candidate_keywords = attribute_dict[selected_category]
    print("=== List of attributes to select ===")
    for idx, keyword in enumerate(candidate_keywords):
        print(f"{idx}. {keyword}")
    selected_attributes = (input("Select attributes: ")) # We assume user input has format like "1,2,3,4"
    selected_keywords_number = [int(attribute) for attribute in selected_attributes.split(",")]
    selected_keywords = [candidate_keywords[num] for num in selected_keywords_number]
    print(f"Selected keywords are: {selected_keywords}")
    return selected_category, selected_keywords

def load_pickle(root, fname):
    with open(os.path.join(root, 'attr_to_img', fname), 'rb') as f:
        att2cloth = pickle.load(f)
        return att2cloth

def ir_find_match(feedback, previous_img, model, index_ids, index_feats, params):
    """
    Input:
        feedback: If initial turn, it would be attributes selected. type: list(str)
        If not, it would be a natural languae feedback given from the user. type: str
        previous_img: Image we recommended at the previous turn.
    Output:
        return imageid of the selected image
    """
    # image examples(from attribute.json files)
    if isinstance(feedback, list):
        # Initial turn
        att2cloth = load_pickle(params['data_root'], '%s_attr_to_imgs.nopickle'%params['category'])
        
        all_clothes = dict()
        for key in feedback:
            matches = att2cloth[key]
            for match in matches:
                if match in all_clothes:
                    all_clothes[match] += 1
                else:
                    all_clothes[match] = 1
        
        max_score = max(all_clothes.items(), key=lambda x: x[1])[1]
        imgs = list()
        for key, score in all_clothes.items():
            if score == max_score:
                imgs.append(key)

        np.random.shuffle(imgs)
        if len(imgs) > 10:
            imgs = imgs[:10]
        
        # NOTE: added to save options
        category = params['category']
        glue_images(imgs, category)

        # match하는걸 보여줌
        # user input 받음
        # 선택된 imageid return
        for i, img in enumerate(imgs):
            print('%d: %s'%(i, img))
        selected = int(input('Please select one item you like the most: '))
        return imgs[selected] # We should return imageid!
    elif isinstance(feedback, str):
        # Turn other than 
        top_k = 10
        if params['category'] == 'bottom':
            new_imgs = find_target_deepfashion(
                
            )
        else:
            new_imgs = find_target(
                            model=model,
                            args=params,
                            index_ids=index_ids,
                            index_feats=index_feats,
                            c_id=previous_img,
                            caption=feedback)
        print(f"Our model searched {top_k} clothes for you")
        
        # NOTE: added to save options
        glue_images(new_imgs, category)
        for img in new_imgs:
            print(f"Image is displayed here: {img}")
        selected = int(input("Please select one item you like the most."))
        return new_imgs[selected]

    else:
        raise ValueError("Invalid feedback type!")





