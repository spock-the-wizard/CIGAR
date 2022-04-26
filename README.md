# Conversational Image Retreival with Virtual Try-On

## Requirements
We assume that you are in the repository directory.
We highly recommend you to create conda environment.
`conda create -n cigar python=3.8 ipython`

### 1. Clone our repository
### 2. Image Retrieval
Our pipeline for image retrieval is folked from https://github.com/nashory/FashionIQChallenge2020.


1. Install all dependencies.
    `pip install -r ir_requirements.txt`
2. Download dataset.
    `cd image_retrieval/retrieval_model/`
    `bash run_download_image.sh`
3. Download word embedding for user feedback.
    `pip install gdown`
    `cd image_retrieval/retrieval_model/ours/train`
    `gdown --fuzzy https://drive.google.com/file/d/1gLl73829eVZrWuXQerJpsz8v9mPQRMAX/view?usp=sharing`
    `tar -xvf assets.tar.gz`
    
4. Download checkpoint for text encoder in model: ask admin (currently not supported)

5. Download FashionIQ dataset

###3. Garment Transfer
Our pipeline for garment transfer is folked from 
1. Parser
- `pip install ninja`
- Download checkpoint

2. Pose Transder
- Install dependency of pose
`cd pose`
`pip install -r requirements.txt`

3. DIOR
- Install dependency of dior
`cd dior`
`pip install -r requirements.txt`
- Create checkpoint, flownet.pth
`gdown --fuzzy <path>`