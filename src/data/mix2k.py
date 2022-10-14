# DIV2K + Flickr2K for lightweight super-resolution

import os
from data import srdata
import glob
from data.div2k import DIV2K
from data.flickr2k import Flickr2K

class MIX2K(srdata.SRData):
    def __init__(self, args, name='MIX2K', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self.DIV2K = DIV2K(args)
        self.Flickr2K = Flickr2K(args)
        self.images_hr = self.DIV2K.images_hr + self.Flickr2K.images_hr
        self.images_lr = []
        for (l, h) in zip(self.DIV2K.images_lr, self.Flickr2K.images_lr):
            self.images_lr.append(l + h)

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)