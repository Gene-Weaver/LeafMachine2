from __future__ import annotations
import torch
import pathlib
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from DocEnTR.models.binae import BINMODEL, BinModel
from einops import rearrange
from tqdm import tqdm
from dataclasses import dataclass, field
import concurrent.futures
from time import perf_counter

''' 
pip install vit-pytorch==0.37.1

### For the 'small x 8 ' version
(venv_LM2) PS D:\Dropbox\LeafMachine2> pip show vit-pytorch
Name: vit-pytorch
Version: 0.37.1
Summary: Vision Transformer (ViT) - Pytorch
Home-page: https://github.com/lucidrains/vit-pytorch
Author: Phil Wang
Author-email: lucidrains@gmail.com
License: MIT
Location: d:\dropbox\leafmachine2\venv_lm2\lib\site-packages
Requires: einops, torch, torchvision
Required-by:
### For the 'base x 16' version
(venv_LM2_38) brlab@brlab-quardo:~/Dropbox/LeafMachine2/leafmachine2/machine/DocEnTR$ pip show vit-pytorch
Name: vit-pytorch
Version: 1.0.2
Summary: Vision Transformer (ViT) - Pytorch
Home-page: https://github.com/lucidrains/vit-pytorch
Author: Phil Wang
Author-email: lucidrains@gmail.com
License: MIT
Location: /home/brlab/Dropbox/LeafMachine2/venv_LM2_38/lib/python3.8/site-packages
Requires: torch, torchvision, einops
Required-by: 

'''
@dataclass
class DocEnTR:
    THRESHOLD: float = 0.50 ## binarization threshold after the model output
    SPLITSIZE: int =  256  ## your image will be divided into patches of 256x256 pixels
    SETTING: str = "small"  ## choose the desired model size [small, base or large], depending on the model you want to use
    patch_size: int = 8 ## choose your desired patch size [8 or 16], depending on the model you want to use
    image_size: list[int] = field(default_factory=list)

    def __init__(self) -> None:
        self.image_size = (self.SPLITSIZE, self.SPLITSIZE)

    def split(self, im,h,w):
        patches=[]
        nsize1=self.SPLITSIZE
        nsize2=self.SPLITSIZE
        for ii in range(0,h,nsize1): #2048
            for iii in range(0,w,nsize2): #1536
                patches.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
        
        return patches 

    def merge_image(self, split_images, h,w):
        image=np.zeros(((h,w,3)))
        nsize1=self.SPLITSIZE
        nsize2=self.SPLITSIZE
        ind =0
        for ii in range(0,h,nsize1):
            for iii in range(0,w,nsize2):
                if ind >= len(split_images):
                    break
                else:
                    image[ii:ii + nsize1, iii:iii + nsize2,:] = split_images[ind]
                    ind += 1
        return image  

    def run_DocEnTR_default(self, cfg, Dirs, dir_to_clean, dir_component):
        binarize_labels_skeletonize = cfg['leafmachine']['cropped_components']['binarize_labels_skeletonize']

        dir_root = os.path.dirname(os.path.dirname(__file__))
        if os.path.exists(dir_to_clean):
            image_list = os.listdir(dir_to_clean)

            # output_dir = dir_component

            device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

            SPLITSIZE = self.SPLITSIZE
            SETTING = self.SETTING
            TPS = self.patch_size

            # batch_size = cfg.batch_size

            experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

            patch_size = TPS
            image_size =  (SPLITSIZE,SPLITSIZE)


            if SETTING == 'base':
                ENCODERLAYERS = 6
                ENCODERHEADS = 8
                ENCODERDIM = 768

            if SETTING == 'small':
                ENCODERLAYERS = 3
                ENCODERHEADS = 4
                ENCODERDIM = 512

            if SETTING == 'large':
                ENCODERLAYERS = 12
                ENCODERHEADS = 16
                ENCODERDIM = 1024

            v = ViT(
                image_size = image_size,
                patch_size = patch_size,
                num_classes = 1000,
                dim = ENCODERDIM,
                depth = ENCODERLAYERS,
                heads = ENCODERHEADS,
                mlp_dim = 2048,
            )

            hyper_params = {"base": [6, 8, 768],
                            "small": [3, 4, 512],
                            "large": [12, 16, 1024]} 

            encoder_layers = hyper_params[SETTING][0]
            encoder_heads = hyper_params[SETTING][1]
            encoder_dim = hyper_params[SETTING][2]

            model = BinModel(
                encoder = v,
                decoder_dim = encoder_dim,      
                decoder_depth = encoder_layers,
                decoder_heads = encoder_heads  
            )

            dir_models = os.path.join(dir_root,'machine','DocEnTR','model_zoo')
            # model_chosen = "best-model_16_2018large_256_16.pt"
            model_chosen = 'small_256_8__epoch-10.pt'
            model_path = os.path.join(dir_models,model_chosen)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            for i, img in tqdm(enumerate(image_list), desc=f'{bcolors.BOLD}     Binarizing images from {dir_to_clean}{bcolors.ENDC}',colour="yellow",position=0,total = len(image_list)):
                # print(f'       Working on image {i}/{n_images} --> {img}')
                deg_image = cv2.imread(os.path.join(dir_to_clean,img)) / 255

                ## Split the image into patches, an image is padded first to make it dividable by the split size
                h =  ((deg_image.shape[0] // 256) +1)*256 
                w =  ((deg_image.shape[1] // 256 ) +1)*256
                deg_image_padded=np.ones((h,w,3))
                deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:] = deg_image
                patches = self.split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])
                ## preprocess the patches (images)
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                out_patches=[]
                for p in patches:
                    out_patch = np.zeros([3, *p.shape[:-1]])
                    for i in range(3):
                        out_patch[i] = (p[:,:,i] - mean[i]) / std[i]
                    out_patches.append(out_patch)

                result = []
                for patch_idx, p in enumerate(out_patches):
                    # print(f"(              {patch_idx} / {len(out_patches) - 1}) processing patch...")
                    p = np.array(p, dtype='float32')
                    train_in = torch.from_numpy(p)

                    with torch.no_grad():
                        train_in = train_in.view(1,3,self.SPLITSIZE,self.SPLITSIZE).to(device)
                        _ = torch.rand((train_in.shape)).to(device)


                        loss,_, pred_pixel_values = model(train_in,_)
                        
                        rec_patches = pred_pixel_values

                        rec_image = torch.squeeze(rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = self.patch_size, p2 = self.patch_size,  h=self.image_size[0]//self.patch_size))
                        
                        impred = rec_image.cpu().numpy()
                        impred = np.transpose(impred, (1, 2, 0))
                        
                        for ch in range(3):
                            impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

                        impred[np.where(impred>1)] = 1
                        impred[np.where(impred<0)] = 0
                    result.append(impred)

                clean_image = self.merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
                clean_image = clean_image[:deg_image.shape[0], :deg_image.shape[1],:]

                clean_image = (clean_image<self.THRESHOLD)*255
                # clean_image = cv2.adaptiveThreshold(clean_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                # clean_image = cv2.bitwise_not(clean_image)
                
                # plt.imshow(clean_image)

                # Step 1: Create an empty skeleton
                clean_image = np.array(clean_image,dtype='uint8')
                
                clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
                
                ret,clean_image = cv2.threshold(clean_image, 127, 255, 0)

                if binarize_labels_skeletonize:
                    try:
                        clean_image = cv2.ximgproc.thinning(clean_image)
                    except:
                        pass

                # model_name = pathlib.Path(model_path).stem
                image_path = pathlib.Path(img)
                # output_path = os.path.join(dir_component,(f'{image_path.stem}__{model_name}{image_path.suffix}'))
                output_path = os.path.join(dir_component,(f'{img}{image_path.suffix}'))

                cv2.imwrite(output_path, clean_image)

    def load_DocEnTR_model(self, device):
        # binarize_labels_skeletonize = cfg['leafmachine']['cropped_components']['binarize_labels_skeletonize']

        model_chosen = 'small_256_8__epoch-10.pt'

        print(f'Loading DocEnTr model: {model_chosen}')

        dir_root = os.path.dirname(os.path.dirname(__file__))

        # output_dir = dir_component

        device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        SPLITSIZE = self.SPLITSIZE
        SETTING = self.SETTING
        TPS = self.patch_size

        # batch_size = cfg.batch_size

        experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

        patch_size = TPS
        image_size =  (SPLITSIZE,SPLITSIZE)


        if SETTING == 'base':
            ENCODERLAYERS = 6
            ENCODERHEADS = 8
            ENCODERDIM = 768

        if SETTING == 'small':
            ENCODERLAYERS = 3
            ENCODERHEADS = 4
            ENCODERDIM = 512

        if SETTING == 'large':
            ENCODERLAYERS = 12
            ENCODERHEADS = 16
            ENCODERDIM = 1024

        v = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1000,
            dim = ENCODERDIM,
            depth = ENCODERLAYERS,
            heads = ENCODERHEADS,
            mlp_dim = 2048
        )

        hyper_params = {"base": [6, 8, 768],
                        "small": [3, 4, 512],
                        "large": [12, 16, 1024]} 

        encoder_layers = hyper_params[SETTING][0]
        encoder_heads = hyper_params[SETTING][1]
        encoder_dim = hyper_params[SETTING][2]

        model = BinModel(
            encoder = v,
            decoder_dim = encoder_dim,      
            decoder_depth = encoder_layers,
            decoder_heads = encoder_heads  
        )

        dir_models = os.path.join(dir_root,'machine','DocEnTR','model_zoo')
        # model_chosen = "best-model_16_2018large_256_16.pt"
        model_path = os.path.join(dir_models,model_chosen)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        return model, device

    def run_DocEnTR_single(self, model, device, img, skeletonize):
    
        # print(f'       Working on image {i}/{n_images} --> {img}')
        deg_image = img / 255

        ## Split the image into patches, an image is padded first to make it dividable by the split size
        h =  ((deg_image.shape[0] // 256) +1)*256 
        w =  ((deg_image.shape[1] // 256 ) +1)*256
        deg_image_padded=np.ones((h,w,3))
        deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:] = deg_image
        patches = self.split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])
        ## preprocess the patches (images)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        out_patches=[]
        for p in patches:
            out_patch = np.zeros([3, *p.shape[:-1]])
            for i in range(3):
                out_patch[i] = (p[:,:,i] - mean[i]) / std[i]
            out_patches.append(out_patch)

        result = []
        for patch_idx, p in enumerate(out_patches):
            # print(f"(              {patch_idx} / {len(out_patches) - 1}) processing patch...")
            p = np.array(p, dtype='float32')
            train_in = torch.from_numpy(p)

            with torch.no_grad():
                train_in = train_in.view(1,3,self.SPLITSIZE,self.SPLITSIZE).to(device)
                _ = torch.rand((train_in.shape)).to(device)


                loss,_, pred_pixel_values = model(train_in,_)
                
                rec_patches = pred_pixel_values

                rec_image = torch.squeeze(rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = self.patch_size, p2 = self.patch_size,  h=self.image_size[0]//self.patch_size))
                
                impred = rec_image.cpu().numpy()
                impred = np.transpose(impred, (1, 2, 0))
                
                for ch in range(3):
                    impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

                impred[np.where(impred>1)] = 1
                impred[np.where(impred<0)] = 0
            result.append(impred)

        clean_image = self.merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
        clean_image = clean_image[:deg_image.shape[0], :deg_image.shape[1],:]

        clean_image = (clean_image<self.THRESHOLD)*255
        # clean_image = cv2.adaptiveThreshold(clean_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # clean_image = cv2.bitwise_not(clean_image)
        
        # plt.imshow(clean_image)

        # Step 1: Create an empty skeleton
        clean_image = np.array(clean_image,dtype='uint8')
        
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
        
        ret,clean_image = cv2.threshold(clean_image, 127, 255, 0)

        if skeletonize:
            clean_image = cv2.ximgproc.thinning(clean_image)

        # model_name = pathlib.Path(model_path).stem
        # image_path = pathlib.Path(img)
        # output_path = os.path.join(dir_component,(f'{image_path.stem}__{model_name}{image_path.suffix}'))
        # output_path = os.path.join(dir_component,(f'{img}{image_path.suffix}'))

        # cv2.imwrite(output_path, clean_image)
        return clean_image

def run_binarize(cfg, logger, Dirs):
    t3_start = perf_counter()
    detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
    logger.name = f'Binarize {detections}'
    logger.info('Binarizing images')
    
    if cfg['leafmachine']['cropped_components']['binarize_labels'] and cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        for component in cfg['leafmachine']['cropped_components']['save_cropped_annotations']:
            if cfg['leafmachine']['cropped_components']['save_per_annotation_class']:
                dir_component = os.path.join(Dirs.save_per_annotation_class,'_'.join([component, 'binary'])) 
                validate_dir(dir_component)
                Labels = DocEnTR()
                dir_to_clean = os.path.join(Dirs.save_per_annotation_class, component)
                Labels.run_DocEnTR_default(cfg, Dirs, dir_to_clean, dir_component)
            if cfg['leafmachine']['cropped_components']['save_per_image']:
                dir_component = os.path.join(Dirs.save_per_image,'_'.join([component, 'binary'])) 
                validate_dir(dir_component)
                Labels = DocEnTR()
                dir_to_clean = os.path.join(Dirs.save_per_image, component)
                Labels.run_DocEnTR_default(cfg, Dirs, dir_to_clean, dir_component)
    t3_stop = perf_counter()
    logger.info(f"[Binarize {detections} elapsed time] {round(t3_stop - t3_start)} seconds")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def validate_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    print('not working')
    # run_DocEnTR()