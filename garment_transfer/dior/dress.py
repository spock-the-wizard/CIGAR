"""Code to Dress up specified garmnet on User image"""

from dior.test_code.pose import *


def dress_up(user,garments):
    user = ("print",1, None)
    garments = [
        ("flower",1,i), # load the 0-th person from "plaid" group, garment #5 (top) is interested
        ("pattern",3,i)
    ]
    import pdb;pdb.set_trace()

    # pose_id == None indicates no pose transfer
    dress_in_order(model,user,gids=garments, order= [],pose_id=None)