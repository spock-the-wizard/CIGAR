"""Code to Dress up specified garmnet on User image"""

from dior.test_code.pose import *


def extract_vton(user_pth,garments,pose_csv):
    """
        user_pth: pth of user image
        garments: list of paths to garment image
    """
    # load img,pose,parsemap for each entry
    
    import pdb;pdb.set_trace()
    load_pose_from_json()

    # pose_id == None indicates no pose transfer
    dress_in_order(model,user,gids=garments, order= [],pose_id=None)