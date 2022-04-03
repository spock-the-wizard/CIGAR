import os

def name_to_filepath(dir,name):
    fnames = []
    for fname in os.listdir(dir):
        if fname.split('.')[-1]==name:
            fnames.append(os.path.join(dir,fname))

    return fnames
    