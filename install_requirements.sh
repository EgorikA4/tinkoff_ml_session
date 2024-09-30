#!/bin/bash

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

# install on CUDA_11.7
pip install wheel
pip install numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install --no-build-isolation -e segmentation/GroundingDINO

# 5. Download models weights
mkdir segmentation/weights

# GroundedSAM weights
wget -P segmentation/weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# GroundingDINO weights
wget -P segmentation/weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# LLava-v1.5 from HF
mkdir recognition/model
huggingface-cli download llava-hf/llava-1.5-7b-hf --cache-dir recognition/model

# OmniFusion from HF
huggingface-cli download AIRI-Institute/OmniFusion models.py OmniMistral-v1_1/projection.pt OmniMistral-v1_1/special_embeddings.pt --local-dir description/
