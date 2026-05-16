from .net import ModelBase
from .cnn import CNNPOS
from .others import Identity, ParamNet
from .code import *

net_dict = {
    'iden': Identity,
    'cnnpos': CNNPOS,
    'codenet': CodeNet,
    'param': ParamNet,
}
