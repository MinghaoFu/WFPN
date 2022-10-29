import torch
from model import *

import utility
import data
import model
import loss
import argparse
import os 

from utils.parse import parse_config
from trainer import Trainer
from thop import clever_format

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--opt', type=str, default='test', help='Name of options JSON file.')

json_path = os.path.join('configs', '{0}.json'.format(parser.parse_args().opt))
args = parse_config(json_path)

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.backends.cudnn.benchmark = True

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()
        print("CUDA max memory allocated: {0}".format(clever_format(torch.cuda.max_memory_allocated(device=torch.cuda.current_device())/ 1024**2)))

if __name__ == '__main__':
    # from utility import test_module
    # test_module(SFPA(16), 16)
    main()
