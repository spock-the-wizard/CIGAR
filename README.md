# Conversational Image Retreival with Virtual Try-On

## Requirements
We assume that you are in the repository directory. \
We highly recommend you to create conda environment. \
`conda create -n cigar python=3.8 ipython`

### 1. Clone our repository
### 2. Image Retrieval
Our pipeline for image retrieval is folked from https://github.com/nashory/FashionIQChallenge2020.


1. Install all dependencies. \
    `pip install -r ir_requirements.txt`
2. Download dataset. \
    `cd image_retrieval/retrieval_model/` \
    `bash run_download_image.sh` \
3. Download word embedding for user feedback. \
    `pip install gdown` \
    `cd image_retrieval/retrieval_model/ours/train` \
    `gdown --fuzzy https://drive.google.com/file/d/1gLl73829eVZrWuXQerJpsz8v9mPQRMAX/view?usp=sharing`
    `tar -xvf assets.tar.gz`
    
4. Download checkpoint for text encoder in model: ask admin (currently not supported)

5. Download FashionIQ dataset

### 3. Garment Transfer
Our pipeline for garment transfer is folked from https://github.com/cuiaiyu/dressing-in-order.git
1. Parser
- `pip install ninja`
- Download checkpoint from [here](https://drive.google.com/drive/folders/11wWszW1kskAyMIGJHBBZzHNKN3os6pu_).

2. Pose Transder
- Install dependencies of pose transfer.
`cd pose`
`pip install -r requirements.txt`

3. Dressing-in-order(DIOR)
- Install dependencies of dior.
`cd dior`
`pip install -r requirements.txt`
- Download checkpoint of pretrained model and unzip at `checkpoints/`
`gdown --fuzzy https://drive.google.com/file/d/1JvLu6RJ4QBAYf6ON9i_DWU3Jlj56vz5P/view?usp=sharing`


## Run
`python main.py`
