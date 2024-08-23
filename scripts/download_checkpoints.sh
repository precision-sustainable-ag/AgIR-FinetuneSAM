#!/bin/bash

echo "Downloading checkpoints..."
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_b_01ec64.pth
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_b_maskdecoder.pth
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_4b8939.pth
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_maskdecoder.pth
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_l_0b3195.pth
wget -P ../pretrained_checkpoint https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_l_maskdecoder.pth
