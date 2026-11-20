import os
os.environ.setdefault('RWKV_ID_FEATURES','1')
os.environ.setdefault('RWKV_REAL_CYCLES','1')
import sys
sys.path.insert(0,'.')
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
m=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
print('features2card_shape',tuple(m.state_dict()['features2card.0.weight'].shape))
