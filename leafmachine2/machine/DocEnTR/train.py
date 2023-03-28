from __future__ import annotations
from typing import Any
import torch
from vit_pytorch import ViT
from models.binae import BINMODEL, BinModel
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from einops import rearrange
import loadData2 as loadData
import utils as utils
from  config import Configs, ConfigsDirect
import os

'''
# ******************************
pip install vit-pytorch==0.37.1
# ******************************
'''
def main():
    # cfg = Configs().parse()
    cfg = ConfigsDirect()
    best_psnr = 0 
    best_epoch = 0

    delete_missing_files(os.path.join(cfg.data_path, 'train'), os.path.join(cfg.data_path, 'train_gt'))
    delete_missing_files(os.path.join(cfg.data_path, 'valid'), os.path.join(cfg.data_path, 'valid_gt'))
    delete_missing_files(os.path.join(cfg.data_path, 'test'), os.path.join(cfg.data_path, 'test_gt'))


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set gpu
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')


    count_psnr = utils.count_psnr
    imvisualize = utils.imvisualize
    load_data_func = loadData.loadData_sets

    transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    SPLITSIZE = cfg.split_size
    SETTING = cfg.vit_model_size
    TPS = cfg.vit_patch_size

    batch_size = cfg.batch_size

    experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

    patch_size = TPS
    image_size =  (SPLITSIZE,SPLITSIZE)

    MASKINGRATIO = 0.5
    VIS_RESULTS = True
    VALID_DIBCO = cfg.validation_dataset

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

    best_psnr = 0
    best_epoch = 0

    trainloader, validloader, testloader = all_data_loader(cfg, load_data_func, batch_size)

    v = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = 1000,
        dim = ENCODERDIM,
        depth = ENCODERLAYERS,
        heads = ENCODERHEADS,
        mlp_dim = 2048
    )

    # model = BINMODEL(
    #     encoder = v,
    #     masking_ratio = MASKINGRATIO,   ## __ doesnt matter for binarization
    #     decoder_dim = ENCODERDIM,      
    #     decoder_depth = ENCODERLAYERS,
    #     decoder_heads = ENCODERHEADS       # anywhere from 1 to 8
    # )
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

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)

    for epoch in range(1,cfg.epochs): 

        running_loss = 0.0
        
        for i, (train_index, train_in, train_out) in enumerate(trainloader):
            
            inputs = train_in.to(device)
            outputs = train_out.to(device)

            optimizer.zero_grad()

            loss, _,_= model(inputs,outputs)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            show_every = int(len(trainloader) / 7)

            if i % show_every == show_every-1:    # print every 20 mini-batches
                print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch, i + 1, running_loss / show_every))
                running_loss = 0.0
        
        
        if VIS_RESULTS:
            if not os.path.exists(os.path.join(cfg.data_path, 'weights')):
                os.makedirs(os.path.join(cfg.data_path, 'weights'))
        
            torch.save(model.state_dict(), os.path.join(cfg.dir_save,''.join([experiment, '__epoch-', str(epoch), '.pt'])))
            # visualize(device, model, str(epoch), validloader, patch_size, image_size, experiment, imvisualize)
            # best_psnr, best_epoch = valid_model(cfg, best_psnr, best_epoch, epoch, model, count_psnr, VALID_DIBCO, experiment, TPS)

'''def visualize(device, model, epoch, validloader, patch_size, image_size, experiment, imvisualize): # Experimental
    losses = 0
    for i, (valid_index, valid_in, valid_out) in enumerate(validloader):
        # inputs, labels = data
        bs = len(valid_in)

        inputs = valid_in.to(device)
        outputs = valid_out.to(device)

        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)
            
            rec_patches = pred_pixel_values

            # add a c dimension with size 3
            rec_patches = torch.unsqueeze(rec_patches, dim=-1).repeat(1, 1, 1, 3)

            rec_images = rearrange(rec_patches, 'b (h p1) (w p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size[0]//patch_size)

            
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(),outputs[j].cpu(),rec_images[j].cpu(),valid_index[j],epoch,experiment)
            
            losses += loss.item()
    
    print('valid loss: ', losses / len(validloader))'''



def visualize(device, model, epoch, validloader, patch_size, image_size, experiment, imvisualize):
    losses = 0
    for i, (valid_index, valid_in, valid_out) in enumerate(validloader):
        # inputs, labels = data could not broadcast input array from shape (3,128,128) into shape (3,)
        bs = len(valid_in)

        inputs = valid_in.to(device)
        outputs = valid_out.to(device)

        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)
            
            rec_patches = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(),outputs[j].cpu(),rec_images[j].cpu(),valid_index[j],epoch,experiment)
            
            losses += loss.item()
    
    print('valid loss: ', losses / len(validloader))

'''def sort_batch(batch):
    train_index = []
    train_in = []
    train_out = []
    for idx, img, gt_img in batch:
        train_index.append(idx)
        train_in.append(img)
        train_out.append(gt_img)

    train_index = np.array(train_index)
    train_in = torch.from_numpy(np.array(train_in)).float()
    train_out = torch.from_numpy(np.array(train_out)).float()

    return train_index, train_in, train_out'''

def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_out = []
    for i in range(n_batch):
        idx, img, gt_img = batch[i]

        train_index.append(idx)
        train_in.append(img)
        train_out.append(gt_img)

    train_index = np.array(train_index)
    train_in = np.array(train_in).astype('float32')
    train_out = np.array(train_out).astype('float32')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)

    return train_index, train_in, train_out


def all_data_loader(cfg, load_data_func, batch_size):
    data_train, data_valid, data_test = load_data_func(cfg)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader

def valid_model(cfg, best_psnr, best_epoch, epoch, model, count_psnr, VALID_DIBCO, experiment, TPS):

    print('last best psnr: ', best_psnr, 'epoch: ', best_epoch)
    
    psnr  = count_psnr(cfg, epoch,valid_data=VALID_DIBCO,setting=experiment)
    print('curr psnr: ', psnr)


    if psnr >= best_psnr:
        best_psnr = psnr
        best_epoch = epoch
        
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
    
        torch.save(model.state_dict(), './weights/best-model_'+str(TPS)+'_'+VALID_DIBCO+experiment+'.pt')

        dellist = os.listdir('vis'+experiment)
        dellist.remove('epoch'+str(epoch))

        for dl in dellist:
            os.system('rm -r vis'+experiment+'/'+dl)
    else:
        os.system('rm -r vis'+experiment+'/epoch'+str(epoch))
    return best_psnr, best_epoch

def delete_missing_files(reference, binary):
    ref_files = set(os.listdir(reference))
    bin_files = set(os.listdir(binary))
    
    missing_files = ref_files - bin_files
    
    for file in missing_files:
        os.remove(os.path.join(reference, file))

if __name__ == '__main__':
    main()