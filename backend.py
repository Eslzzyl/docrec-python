import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

from seg import U2NETP
from GeoTr import GeoTr
from ill_rec import rec_ill

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        print('Sementation working...', end='')
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x
        print('Done.')
        print('GeoTr working...', end='')
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm

def reload_model(model: GeoTr_Seg, path='./model_pretrained/') -> GeoTr_Seg:
    seg_model_dict = model.msk.state_dict()
    seg_pretrained_dict = torch.load(path + 'seg.pth', map_location='cpu')
    # print(len(seg_pretrained_dict.keys()))
    print('Segmentation model successfully reloaded.')
    seg_pretrained_dict = {k[6:]: v for k, v in seg_pretrained_dict.items() if k[6:] in seg_model_dict}
    # print(len(seg_pretrained_dict.keys()))
    seg_model_dict.update(seg_pretrained_dict)
    model.msk.load_state_dict(seg_model_dict)

    geo_model_dict = model.GeoTr.state_dict()
    geo_pretrained_dict = torch.load(path + 'geotr.pth', map_location='cpu')
    # print(len(geo_pretrained_dict.keys()))
    print('GeoTr model successfully reloaded.')
    geo_pretrained_dict = {k[7:]: v for k, v in geo_pretrained_dict.items() if k[7:] in geo_model_dict}
    # print(len(geo_pretrained_dict.keys()))
    geo_model_dict.update(geo_pretrained_dict)
    model.GeoTr.load_state_dict(geo_model_dict)

    return model
