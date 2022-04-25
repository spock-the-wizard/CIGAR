#!/bin/bash
python main.py \
    --warmup \
    --expr_name 'bert_0' \
    —-batch_size 32 \
    --caption_directory '/home/deokhk/coursework/CIGAR/data/image_retrieval/generated_captions' \
    --caption_file_name 'augmented_0.0_bert.json' \
    --data_root '/home/deokhk/coursework/' \
    --test_root '/home/deokhk/coursework/CIGAR/image_retrieval/retrieval_model/ours/train' \
    --target 'lower'