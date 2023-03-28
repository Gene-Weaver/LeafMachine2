import os, random
import shutil

def copy_n_images_to_new_dir(dir_source, dir_destination, n_sample): 
    # dir_source = '/media/LMDB/Icacinaceae_Database/CVH_Images'
    # dir_destination = '/home/brlab/Dropbox/LM2_Env/Image_Datasets/CVH_Icacinaceae_500'
    # n_sample = 500

    n_max = len(os.listdir(dir_source))
    if 0 < n_sample < 1:
        n_sample = int(n_sample * n_max)
        n_rand = random.sample(range(0, n_max), n_sample)
    else:
        n_rand = random.sample(range(0, n_max), n_sample)

    for ind, image in enumerate(os.listdir(dir)):
        if ind in n_rand:
            src = os.path.join(dir_source, image)
            dst = os.path.join(dir_destination, image)
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    copy_n_images_to_new_dir('/media/LMDB/Icacinaceae_Database/CVH_Images', '/home/brlab/Dropbox/LM2_Env/Image_Datasets/CVH_Icacinaceae_500', 500)