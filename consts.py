"""Module that provides constant variables."""

DEVICE = 'cuda'

LLAVA_REPO_ID = 'llava-hf/llava-1.5-7b-hf'
LLAVA_MAX_NEW_TOKENS = 200
LLAVA_CACHE_DIR = 'recognition/model'

DINO_CONFIG_PATH = 'segmentation/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
DINO_CKPT_PATH = 'segmentation/weights/groundingdino_swint_ogc.pth'

SAM_CKPT_PATH = 'segmentation/weights/sam_vit_h_4b8939.pth'

OMNIFUSION_REPO_ID = 'AIRI-Institute/OmniFusion'
OMNIFUSION_PROJECTION = 'description/OmniMistral-v1_1/projection.pt'
OMNIFUSION_EMBEDDINGS = 'description/OmniMistral-v1_1/special_embeddings.pt'

ITEMS_RECOGNITION_CONVERSATION = [
    {
        'role': 'user',
        'content': [
            {'type': 'image'},
            {
                'type': 'text',
                'text': 'What category of products do you see in the picture? Only their names.',
            },
        ],
    },
]

GROUNDING_DINO_PROMPT = 'detect all items that according to category of {0} without background.'

OMNIFUSION_PROMPT = 'This is a dialogue with an AI assistant who perfectly writes a description for the sale of goods.\n'
OMNIFUSION_QUERY = 'Напиши на русском информативное и привлекательное описание товара.'
