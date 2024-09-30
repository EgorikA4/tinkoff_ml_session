#!/bin/bash

# TODO: install GROUNDEDSAM and extraction only GroundingDINO and segment anything
# also add in the README file that I've use cuda11.7 so I need to install some juice...
# create blur on image ... preprocessing and other stuff. Total weights -> 15.2GB OmniFusion + 13GB GD + SAM + quantizedLLava = 28.2GB all models

# добавить cache-dir для скачивания с hf


# 1. Cloning Grounded-Segment-Anything into 'segmentation' folder
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git segmentation/Grounded-Segment-Anything
# 2. Extract GroundingDINO and segment-anything folder from Grounded-Segment-Anything
mv segmentation/Grounded-Segment-Anything/GroundingDINO segmentation/GroundingDINO
mv segmentation/Grounded-Segment-Anything/segment_anything segmentation/segment_anything
mv segmentation/Grounded-Segment-Anything/requirements.txt segmentation/requirements.txt
# 3. Remove Grounded-Segment-Anything folder
rm -rf segmentation/Grounded-Segment-Anything

# 4. Install python requirements
pip install -r requirements/main.txt
pip install -r segmentation/requirements.txt
pip install -e segmentation/segment_anything
pip install --no-build-isolation -e segmentation/GroundingDINO

# 5. Download models weights
mkdir segmentation/weights

# GroundedSAM weights
wget segmentation/weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# GroundingDINO weights
wget segmentation/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# LLava-v1.5 from HF
mkdir recognition/model
huggingface-cli download --cache-dir recognition/model llava-hf/llava-1.5-7b-hf

# OmniFusion from HF
# TODO: --cache-dir or --local-dir ?? './' by default
huggingface-cli download AIRI-Institute/OmniFusion models.py OmniMistral-v1_1/projection.pt OmniMistral-v1_1/special_embeddings.pt

# pip install wheel
# pip install torch torchvision torchauio --index-url https://download.pytorch.org/whl/cu117
# If CUDA 11.7
