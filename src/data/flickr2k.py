import os
from data import srdata

class Flickr2K(srdata.SRData):
    def __init__(self, args, name='Flickr2K', train=True, benchmark=False):
        super(Flickr2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(Flickr2K, self)._scan()
        return names_hr, names_lr
        #return names_hr[0:2400], [n[0:2400] for n in names_lr]

    def _set_filesystem(self, dir_data):
        super(Flickr2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Flickr2K_HR')
        self.dir_lr = os.path.join(self.apath, 'Flickr2K_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

