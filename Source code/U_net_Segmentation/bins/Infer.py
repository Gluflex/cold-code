from fastai.vision.all import *
from matplotlib import pyplot as plt
from time import sleep
import cv2 as cv
from math import sqrt
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

path = Path('D:\\starthack_redbull')
path_im = path/'im'
path_lbl = path/'gnd'
valid_fnames = (path/'validation.txt').read_text().split('\n')
classes = np.loadtxt(path/'classes.txt', dtype=str)
name2id = {v:k for k,v in enumerate(classes)}
void_code = name2id['_background_']


def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read_text().split('\n') 
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner

def acc_func(src, target):
  target = target.squeeze(1)
  mask = target != void_code
  return (src.argmax(dim=1)[mask]==target[mask]).float().mean()

def getMask(img):
    return path_lbl/f'gnd_{img.stem}.png'

def write2File(msg:str,filePath = './data.txt'):
   '''whats detected, coordinates,size'''
   with open(filePath,'w') as fp:
      fp.write(msg)

def blobDetector(img,prod):
   '''
   prod = 1 = redbull
   prod = 2 = Coca cola
   
   '''
   size =  np.count_nonzero(img==prod)
   print(np.amax(img))
   plt.imshow(img)
   plt.show()
   if size>7000:
      pos = np.mean(np.where(img>0),axis=1)
      return size,pos
   else:
      return None,None
    

if __name__ == '__main__':
    
    fnames = get_image_files(path_im)
    lbl_names = get_image_files(path_lbl)
    datbBlock = DataBlock(blocks=(ImageBlock, MaskBlock(classes)),get_items=get_image_files,splitter=FileSplitter(path/'validation.txt'),get_y=getMask,item_tfms=Resize(460), batch_tfms=aug_transforms().append(Normalize.from_stats(*imagenet_stats)))
    dls = datbBlock.dataloaders(path_im,bs = 1, num_workers=0)
    opt = ranger
    learn = unet_learner(dls, resnet34, metrics=acc_func, self_attention=True, act_cls=Mish, opt_func=opt,pretrained= True)
    lr = 1e-3
    lrs = slice(lr/20)
    learn.load(path/'BullNetmax')
    while True:
        try:
            dl = learn.dls.test_dl(get_image_files(path/'test'))
        except PermissionError:
           sleep(0.2)
        preds = learn.get_preds(dl=dl)
        pred_1 = preds[0][0]
        pred_arx = pred_1.argmax(dim=0)
        pred_arx = pred_arx.numpy()
        rescaled = (500 / pred_arx.max() * (pred_arx - pred_arx.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)

        rbsize,rbpos = blobDetector(pred_arx,1)
        ccsize,ccpos = blobDetector(pred_arx,2)
        
        msg = 'RedBull,'+str(rbsize)+','+str(rbpos)+'\n'+'Cola,'+str(ccsize)+','+str(ccpos)
        print(msg)
        write2File(msg)
    plt.imshow(im)
    plt.show()

    
