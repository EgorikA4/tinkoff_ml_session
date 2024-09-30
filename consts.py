"""Module that provides constant variables."""

DEVICE = 'cuda'

LLAVA_REPO_ID = 'llava-hf/llava-1.5-7b-hf'
LLAVA_MAX_NEW_TOKENS = 200
LLAVA_CACHE_DIR = '/home/tinkoff_session/recognition/models'

DINO_CONFIG_PATH = '/home/tinkoff_session/segmentation/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
DINO_CKPT_PATH = '/home/tinkoff_session/segmentation/models/weights/groundingdino_swint_ogc.pth'

SAM_CKPT_PATH = '/home/tinkoff_session/segmentation/models/weights/sam_vit_h_4b8939.pth'

OMNIFUSION_REPO_ID = 'AIRI-Institute/OmniFusion'


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


OMNIFUSION_PROMPT = ''
OMNIFUSION_QUERY = ''

GROUNDING_DINO_PROMPT = 'find items that according to category of {0}'
