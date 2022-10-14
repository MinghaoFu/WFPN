import os
from datetime import datetime
import json
import warnings
import torch.nn as nn
import torch

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def get_json(file_path):
    json_str = ''
    with open(file_path, 'r') as f:
        for line in f:
            # remove comments starting with '//'
            line = line.split('//')[0] + '\n'
            json_str += line
    json_t = json.loads(json_str)

    return json_t

def parse_config(opt_path, default_path=os.path.join('configs', 'plain.json')):
    ext = opt_path.split('.')[-1]
    if ext == 'json':
        opt = get_json(default_path)
        if os.path.exists(opt_path):
            model_opt = get_json(opt_path)
            opt.update(model_opt)
    # if other extension...
    else:
        raise ValueError(
            'Option file extention is not supported.'
        )
    opt = Params(opt)
    setattr(opt, 'timestamp', get_timestamp())
    setattr(opt, 'model', opt_path.split('.')[-2].split('/')[-1].upper())

    if not hasattr(opt, 'save'): 
        opt.save = opt.model # relative save path

    opt.betas = tuple(opt.betas)
    opt.data_train = opt.data_train.split('+')
    opt.data_test = opt.data_test.split('+')
    opt.scale = [int(scale) for scale in opt.scale.split('+')]
    set_template(opt)

    
    return opt

def flat_dict(r_dict, f_dict):
    for cate, value in r_dict.items():
        if isinstance(value, dict):
            flat_dict(value, f_dict)
        else:
            f_dict[cate] = value

class Params:
    def __init__(self, params_dict):
        f_dict = {}
        flat_dict(params_dict, f_dict)
        for val in f_dict.values():
            if not (
                isinstance(val, int)
                or isinstance(val, float)
                or isinstance(val, str)
                or isinstance(val, list)
            ):
                raise ValueError(
                    "Parameter value {} should be integer, float, string or list.".format(
                        val
                    )
                )
        self._values = {}
        for param in f_dict:
            setattr(self, param, f_dict[param])

    def __repr__(self):
        return "Params object with values {}".format(self._values.__repr__())

    def __setattr__(self, __name, __value) -> None:
        self.__dict__[__name] = __value
        self._values[__name] = __value

    def values(self):
        return self._values

def set_template(args):
    if args.template in ['M4C8', 'M4C16', 'M10C16', 'M10C32', 'M16C64']:
        mi = args.template.index('M')
        ci = args.template.index('C')
        setattr(args, 'n_feats', int(args.template[ci+1:]))
        setattr(args, 'n_blocks', int(args.template[mi+1:ci]))
        print('===>Template: M{0}C{1}'.format(args.n_blocks, args.n_feats))
