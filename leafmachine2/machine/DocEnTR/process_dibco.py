import numpy as np
import os 
from  PIL import Image
import random
import cv2
from shutil import copy
from tqdm import tqdm
from config  import Configs

cfg = Configs().parse()

main_path = cfg.data_path
validation_dataset = cfg.validation_dataset
testing_dataset = cfg.testing_dataset

def prepare_dibco_experiment(val_set,test_set, patches_size, overlap_size, patches_size_valid):
    
    folder = main_path+'DIBCOSETS/'

    all_datasets = os.listdir(folder)
    n_i = 1

    for d_set in tqdm(all_datasets):
        if d_set not in  [val_set,test_set]:
            # continue
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],overlap_size):
                    for j in range (0,img.shape[1],overlap_size):
                        try:
                            if i+patches_size<=img.shape[0] and j+patches_size<=img.shape[1]:
                                p = img[i:i+patches_size,j:j+patches_size,:]
                                gt_p = gt_img[i:i+patches_size,j:j+patches_size,:]
                            
                            elif i+patches_size>img.shape[0] and j+patches_size<=img.shape[1]:
                                p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                                gt_p = np.ones((patches_size,patches_size,3)) *255
                                
                                p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size,:]
                                gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size,:]
                            
                            elif i+patches_size<=img.shape[0] and j+patches_size>img.shape[1]:
                                p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                                gt_p = np.ones((patches_size,patches_size,3)) * 255
                                
                                p[:,0:img.shape[1]-j,:] = img[i:i+patches_size,j:img.shape[1],:]
                                gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size,j:img.shape[1],:]

                            else:
                                p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                                gt_p = np.ones((patches_size,patches_size,3)) * 255
                                
                                p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                                gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]


                            
                            cv2.imwrite(main_path+'train/'+str(n_i)+'.png',p)
                            cv2.imwrite(main_path+'train_gt/'+str(n_i)+'.png',gt_p)
                            n_i+=1
                        except:
                            n_i+=1
                            pass
        if d_set == test_set:
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],patches_size_valid):
                    for j in range (0,img.shape[1],patches_size_valid):
                        
                        try:
                            if i+patches_size_valid<=img.shape[0] and j+patches_size_valid<=img.shape[1]:
                                p = img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                                gt_p = gt_img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                            
                            elif i+patches_size_valid>img.shape[0] and j+patches_size_valid<=img.shape[1]:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                                
                                p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size_valid,:]
                                gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size_valid,:]
                            
                            elif i+patches_size_valid<=img.shape[0] and j+patches_size_valid>img.shape[1]:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                
                                p[:,0:img.shape[1]-j,:] = img[i:i+patches_size_valid,j:img.shape[1],:]
                                # gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size_valid,j:img.shape[1],:]
                                gt_p[:, 0:min(img.shape[1]-j, gt_img.shape[1]-j), :] = gt_img[i:i+patches_size_valid, j:j+min(patches_size_valid, gt_img.shape[1]-j), :]

                            else:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                
                                p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                                gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]

                
                            cv2.imwrite(main_path+'test/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                            cv2.imwrite(main_path+'test_gt/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)
                        except:
                            pass

        if d_set == val_set:
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],patches_size_valid):
                    for j in range (0,img.shape[1],patches_size_valid):

                        try:

                            if i+patches_size_valid<=img.shape[0] and j+patches_size_valid<=img.shape[1]:
                                p = img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                                gt_p = gt_img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                            
                            elif i+patches_size_valid>img.shape[0] and j+patches_size_valid<=img.shape[1]:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                                
                                p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size_valid,:]
                                gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size_valid,:]
                            
                            elif i+patches_size_valid<=img.shape[0] and j+patches_size_valid>img.shape[1]:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                
                                p[:,0:img.shape[1]-j,:] = img[i:i+patches_size_valid,j:img.shape[1],:]
                                gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size_valid,j:img.shape[1],:]

                            else:
                                p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                                
                                p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                                gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]

                
                            cv2.imwrite(main_path+'valid/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                            cv2.imwrite(main_path+'valid_gt/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)
                        except:
                            pass

if not os.path.exists(main_path+'train/'):
    os.makedirs(main_path+'train/')
if not os.path.exists(main_path+'train_gt/'):
    os.makedirs(main_path+'train_gt/')

if not os.path.exists(main_path+'valid/'):
    os.makedirs(main_path+'valid/')
if not os.path.exists(main_path+'valid_gt/'):
    os.makedirs(main_path+'valid_gt/')

if not os.path.exists(main_path+'test/'):
    os.makedirs(main_path+'test/')
if not os.path.exists(main_path+'test_gt/'):
    os.makedirs(main_path+'test_gt/')
    


os.system('rm '+main_path+'train/*')
os.system('rm '+main_path+'train_gt/*')
                
os.system('rm '+main_path+'valid/*')
os.system('rm '+main_path+'valid_gt/*')

os.system('rm '+main_path+'test/*')
os.system('rm '+main_path+'test_gt/*')

patch_size =  cfg.split_size

p_size = (patch_size+128)
p_size_valid  = patch_size
overlap_size = patch_size//2

prepare_dibco_experiment(validation_dataset, testing_dataset,p_size,overlap_size,p_size_valid)



exit(0)
