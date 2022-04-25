import numpy as np
from torch import manual_seed
from image_retrieval.retrieval_model.ours.tools import find_target
def ir_show_keywords():
    """
    Input: X
    Output: 
        selected_keywords / list(str)

    In this function, we should show 
    """
    categories = ['dress', 'shirt', 'toptee', 'pants']
    for idx, keyword in enumerate(categories):
        print(f"{idx}. {keyword}")
    category_idx = (input("Select one category: ")) # We assume user input has format like "1,2,3,4"
    selected_category = categories[category_idx]
    candidate_keywords = ["leaf", "leather", "leather mini", "leather skater", "leopard"]
    print("=== List of attributes to select ===")
    for idx, keyword in enumerate(candidate_keywords):
        print(f"{idx}. {keyword}")
    selected_attributes = (input("Select attributes: ")) # We assume user input has format like "1,2,3,4"
    selected_keywords_number = [int(attribute) for attribute in selected_attributes.split(",")]
    selected_keywords = [candidate_keywords[num] for num in selected_keywords_number]
    print(f"Selected keywords are: {selected_keywords}")
    return selected_category, selected_keywords

def ir_find_match(feedback, previous_img, category, model, index_ids, index_feats, params):
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
        att2cloth = dict() 
        att2cloth['halter'] = ['A','B','C']
        att2cloth['dress'] = ['A', 'C']
        att2cloth['leopard'] = ['D', 'E']
        att2cloth['leather skater'] = ['C', 'A']
        att2cloth['leather mini'] = ['C', 'A']
        att2cloth['leaf'] = ['B', 'C', 'D']
        att2cloth['leather'] = ['B', 'C', 'D']

        all_clothes = ['A','B','C','D','E']
        clothes = {k:0 for k in all_clothes}
        for key in feedback:
            matches = att2cloth[key]
            for match in matches:
                clothes[match] += 1
        max_score = max(clothes.items(), key=lambda x: x[1])[1]
        imgs = list()
        for key, score in clothes.items():
            if score == max_score:
                imgs.append(key)
        # match하는걸 보여줌
        # user input 받음
        # 선택된 imageid return
        for img in imgs:
            print(img)
        selected = int(input('Please select one item you like the most'))
        return  imgs[selected]# We should return imageid!
    elif isinstance(feedback, str):
        # Turn other than 
        top_k = 10
        new_imgs = find_target(gpu_id=params.gpu_id,
                        manualseed=params.manualSeed,
                        data_root=params.data_root,
                        test_root=params.test_root,
                        img_size=params.img_size,
                        model = model,
                        index_ids=index_ids,
                        index_feats=index_feats,
                        c_id=previous_img,
                        category= category,
                        caption= feedback)
        print(f"Our model searched {top_k} clothes for you")
        for img in new_imgs:
            print(f"Image is displayed here: {img}")
        selected = int(input("Please select one item you like the most."))
        return new_imgs[selected]

    else:
        raise ValueError("Invalid feedback type!")





