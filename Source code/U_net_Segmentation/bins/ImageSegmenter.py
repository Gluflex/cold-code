from fastai.vision.all import *
from matplotlib import pyplot as plt
import cv2 as cv


path = Path('D:\\starthack_redbull')
path_im = path/'im'
path_lbl = path/'gnd'
valid_fnames = (path/'validation.txt').read_text().split('\n')
classes = np.loadtxt(path/'classes.txt', dtype=str)
name2id = {v:k for k,v in enumerate(classes)}
void_code = name2id['_background_']

#class CrossEntropyLossFlat(BaseLoss):
#    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
#    y_int = True
#    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
#    def decodes(self, x):    return x.argmax(dim=self.axis)
#    def activation(self, x): return F.softmax(x, dim=self.axis)

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

if __name__ == '__main__':
    
    fnames = get_image_files(path_im)
    lbl_names = get_image_files(path_lbl)
 
    img_fn = fnames[2]
    msk =cv.imread(str(getMask(img_fn)))
    clamp =cv.imread(str(img_fn))
    sz = msk.shape
    full = tuple(int(x) for x in sz)
    datbBlock = DataBlock(blocks=(ImageBlock, MaskBlock(classes)),get_items=get_image_files,splitter=FileSplitter(path/'validation.txt'),get_y=getMask,item_tfms=Resize(460), batch_tfms=aug_transforms().append(Normalize.from_stats(*imagenet_stats)))
    dls = datbBlock.dataloaders(path_im,bs = 1, num_workers=0)
    opt = ranger
    #clamp = clamp*msk
    # Create learner with resnet34 architecture
    #plt.imshow(clamp)
    #plt.show()
    learn = unet_learner(dls, resnet101, metrics=acc_func, self_attention=True, act_cls=Mish, opt_func=opt,pretrained= True)
    #learn.save('model_Lodd_NoPre_Res34')
    lr = 1e-3
    lrs = slice(lr/20)

########## Training ########################################################
    
    learn.fine_tune(20)
    learn.save(path/'BullNetMax')
    learn.load(path/'BullNetMax')
    #learn.save(path/'modelLoddPre')
    #learn.predict(path/'weird.png')
    #learn.show_results()
    #learn.unfreeze()
    #lr = 1e-5
    #lrs = slice(lr/20)
    #learn.fit_flat_cos(20, lrs)
    #learn.save(path/'clampNet_fine')
    #learn.show_training_loop
    
    #learn.show_results(max_n=4, figsize=(18,8), shuffle= True)
    #plt.show()
    #learn.loss_func = CrossEntropyLossFlat(weight=weights, axis=1)
    #learn.save('modelLoddPre2')

########## Training ########################################################
    
    dl = learn.dls.test_dl(get_image_files(path/'test'))
    preds = learn.get_preds(dl=dl)
    pred_1 = preds[0][0]
    pred_arx = pred_1.argmax(dim=0)
    pred_arx = pred_arx.numpy()
    rescaled = (500 / pred_arx.max() * (pred_arx - pred_arx.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    plt.imshow(im)
    #learn.show_results()
    plt.show()
    #for i, pred in enumerate(preds[0]):
    #    pred_arg = pred.argmax(dim=0).numpy()
    #    rescaled = (255.0 / pred_arg.max() * (pred_arg - pred_arg.min())).astype(np.uint8)
    #    im = Image.fromarray(rescaled)
    #    im.save(path/f'Image_{i}.png')
    #torch.save(preds[0][0], 'Image_1.pt')
    #pred_1 = torch.load('Image_1.pt')
    #test = cv.imread(str(get_image_files(path/'test')[0]),0)
    #cv.addWeighted(test, 0.3, np.array(pred_1.argmax(dim=0)), 0.7, 0, test)
    #plt.imshow(test)
    #plt.show()
