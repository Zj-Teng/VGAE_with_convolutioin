import abc

import torch


class _MetaConst:
    __metaclass__ = abc.ABCMeta

    def __setattr__(self, key, value):
        pass


class _Config(_MetaConst):
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8
    EPOCH = 400
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 5e-4

    def __init__(self):
        super(_Config, self).__init__()

    def __setattr__(self, key, value):
        if key in _Config.__dict__:
            raise _AttributeAccessError('Do not rebind {}, it is a constant value'.format(key))
        if not key.upper():
            raise _ConstNameError('Constant value {} must be uppercase'.format(key))


class _ConstNameError(Exception):

    def __init__(self, msg):
        super(_ConstNameError, self).__init__(msg)


class _AttributeAccessError(Exception):

    def __init__(self, msg):
        super(_AttributeAccessError, self).__init__(msg)


CONFIG = _Config()
