from argparse import Namespace
from easydict import EasyDict


def as_easy_dict(dic):
    if isinstance(dic, dict):
        return EasyDict(dic)
    elif isinstance(dic, Namespace):
        return dic
    else:
        raise TypeError("The type must be `dict` or `argparse.Namespace`")
