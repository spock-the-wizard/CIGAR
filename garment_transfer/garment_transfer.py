

# def run_parser(model, input_dir, output_dir):
#     """
#     SCHP의 simple_extractor.py 실행
#     Input: input_dir = garment image path
#     Output: 코드에서는 parse된 이미지와 logits_result가 저장됨. 
#     -> path를 return할 것인가 array를 return할 것인가?

#     """
#     par_img = 1
#     return par_gar_img

# def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2], perturb=False):
#     """
#     Input:
#     - dress_in_order model code
#     - model: DIOR모델
#     - pid: person id, tuple (filename, None, None)
#     - pose_id: 
#     - gids: garments to try on, list of tuples
#     - ogids: garments to lay over, list of tuples
#     - order: , list
#     - perturb: , boolean
#     """
#     pimg = 1
#     gimgs = 1
#     oimgs = 1
#     gen_img = [1,2,3]
#     to_pose = 1
#     return pimg, gimgs, oimgs, gen_img[0], to_pose

def gt_generate(user_img, gar_img, parse_output_dir='data/'):
    """
    Input: 
    - user_img: 
    - gar_img: garment image path.
    Output: TryOn Image

    """
    print('==========GARMENT TRANSFER=============')

    
    # parse garment image
    # parsed_gar_img = run_parser(gar_img, parse_output_dir)

    # 지금 가진 정보로 dior에 필요한 input을 어떻게 얻을 것인가??

    # create model
    # model = 1 #DIORModel(args)
    #model.setup(args)

    # set input
    # person id
    # pid = 1 #('userfile.png', None, None)
    # garments to try on
    # ogids = [1,2] #[('print', 2, 5)]
    
    # run DIOR
    # pimg, gimgs, oimgs, gen_img[0], to_pose = dress_in_order(model, pid, ogids=ogids)
    print('~computing vton image~')

    # show result
    print('--------------------------')#This is the vton result image')
    print('|                         |')
    print('|                         |')
    print('|       VTON res          |')
    print('|                         |')
    print('|                         |')
    print('--------------------------')
    # save result
    print('Saving the image...')

    # if plot:
    #     plt.imshow(oimgs) 
    # else:
    #     return oimgs
    
    
