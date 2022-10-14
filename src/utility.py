from filecmp import dircmp
import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
from tkinter import W
from pytorch_msssim import ssim
from thop import profile, clever_format
from scipy import signal
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler  
from scipy import signal
import torchvision.transforms as transforms
import matplotlib
import glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        if self.args.ssim:
            self.ssim = torch.zeros(1,5,1)
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save_config(self, model):
        with open(self.get_path('config.txt'), 'a') as f:
            f.write(model.__repr__())
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
    # def calculate_psnr(img1, img2, border=0):
    #     # img1 and img2 have range [0, 255]
    # if not img1.shape == img2.shape:
    #     raise ValueError('Input images must have the same dimensions.')
    # h, w = img1.shape[:2]
    # img1 = img1[border:h-border, border:w-border]
    # img2 = img2[border:h-border, border:w-border]

    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2)
    # if mse == 0:
    #     return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

# def calc_ssim(X, Y, scale, rgb_range, dataset=None, sigma=1.5, K1=0.01, K2=0.03, R=255):
#   gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

#   if dataset and dataset.dataset.benchmark:
#     shave = scale
#     if X.size(1) > 1:
#         gray_coeffs = [65.738, 129.057, 25.064]
#         convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#         X = X.mul(convert).sum(dim=1)
#         Y = Y.mul(convert).sum(dim=1)
#   else:
#     shave = scale + 6

#   X = X[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64) 
#   Y = Y[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)

#   window = gaussian_filter

#   ux = signal.convolve2d(X, window, mode='same', boundary='symm')
#   uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

#   uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
#   uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
#   uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

#   vx = uxx - ux * ux
#   vy = uyy - uy * uy
#   vxy = uxy - ux * uy

#   C1 = (K1 * R) ** 2
#   C2 = (K2 * R) ** 2

#   A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
#   D = B1 * B2
#   S = (A1 * A2) / D
#   mssim = S.mean()

#   return mssim
def calc_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    # '''
    # if not img1.shape == img2.shape:
    #     raise ValueError('Input images must have the same dimensions.')
    
    # if img1.ndim == 2:
    #     return ssim(img1, img2)
    # elif img1.ndim == 4:
    #     ssims = []
    #     for i in range(3):
    #         ssims.append(ssim(img1[:, i:i+1, :, :].cpu(), img2[:, i:i+1, :, :].cpu()))
    #     return np.array(ssims).mean()
    # else:
    #     raise ValueError('Wrong input dims in calc_ssim')
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    if args.decay_type == 'multi step':
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR
    elif args.decay_type == 'step':
        kwargs_scheduler = {'step_size': args.step_size, 'gamma': args.gamma}
        scheduler_class = lrs.StepLR
    elif args.decay_type == 'cosine':
        T_period = [250, 500, 750, 1000, 1250]
        restarts = [250, 500, 750, 1000, 1250]
        restart_weights = [1, 1, 1, 1, 1]

        kwargs_scheduler = {'T_period': T_period, 'eta_min': 1e-7, 'restarts': restarts, 'weights': restart_weights}
        scheduler_class = CosineAnnealingLR_Restart


    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
    
def model_profile(model, input=torch.randn(1, 3, 192, 192)):
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, params = profile(model, inputs=(input, ))
    # params = float('{:.3g}'.format(params))
    # magnitude = 0
    # while abs(params) >= 1000:
    #     magnitude += 1
    #     params /= 1000.0
    # params = '{}{}'.format('{:f}'.format(params).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
    macs, params = clever_format([macs, params], "%.3f")
    print('===>Model parameters in training: {0}'.format(params))
    print('===>Macs: {0}'.format(macs))

def RGB2YChannel(T):
    """T: [B, 3, H, W]
    """
    return T[:, 0:1, :, :] * 0.299 + T[:, 1:2, :, :] * 0.587 + T[:, 2:3, :, :] * 0.114

def crop_comparison(dir_path, left, top, right, bottom, name='crop', scale=4):
    """Generate different model comparison through crop the same region
    Args:
        save_path (list): different model results
        scale (int): scale factor
    """
    save_dir = os.path.join(dir_path, name)
    os.makedirs(save_dir)
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            save_file = os.path.join(save_dir, file)
            img = Image.open(os.path.join(dir_path, file))
            if file[0:2] == 'LR':
                img = img.crop((left // 4, top // 4, right // 4, bottom // 4))
            else:
                img = img.crop((left, top, right, bottom))
            img.save(save_file)

def patch_test(*imgs):
    patch = []
    for img in imgs:
        patch.append(img[:, :, ])

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'cosine':
        # cosine annealing restart
        ## need more to prevent error
        T_period = [250, 500, 750, 1000, 1250]
        restarts = [250, 500, 750, 1000, 1250]
        restart_weights = [1, 1, 1, 1, 1]

        scheduler = CosineAnnealingLR_Restart(my_optimizer, T_period, eta_min=1e-7, restarts=restarts,
                                            weights=restart_weights)

    return scheduler

class CosineAnnealingLR_Restart(_LRScheduler):
    """
    ref:https://github.com/zhaohengyuan1/PAN/blob/a20974545cf011c386d728739d091c39e23d0686/codes/models/lr_scheduler.py
    ref: pytorch_CosineAnnealingLR_doc  https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

def plot_comp(epoch, save_path, logs, labels, dim):
    axis = np.linspace(1, epoch, epoch)
    #label = 'Wider Feature Projection Network (WFPN) without attention'
    fig = plt.figure()
    #plt.title(label)
    for log, label in zip(logs, labels):
        if dim == 2:
            log = log[:, 0].numpy()[0:epoch]
        elif dim == 3:
            log = log[:, 0, 0].numpy()[0:epoch]
        plt.plot(
            axis,
            log,
            label=label
        )
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close(fig)