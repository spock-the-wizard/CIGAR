#!/bin/bash
python main.py \
    --warmup \
    —-batch_size 16 \
    --caption_directory '/home/deokhk/coursework/CIGAR/data/image_retrieval/generated_captions' \
    --caption_file_name 'train_short.json' \
    --data_root '/home/deokhk/coursework/' \
    --test_root '/home/deokhk/coursework/FashionIQChallenge2020/ours/train' \
    --target 'lower'