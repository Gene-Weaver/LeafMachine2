import imp, os, inspect, sys
import pandas as pd 
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import validate_dir
from leafmachine2.machine.utils_GBIF import download_subset_images_user_file, check_n_images_in_group, download_subset_images_user_file_no_wishlist, get_cfg_from_full_path




#######################################################################
########################## Fagaceae - Hipp ############################
#######################################################################
dir_LM2_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dir_LM2_configs = os.path.join(dir_LM2_home,'configs')
path_cfg = os.path.join(dir_LM2_configs,'config_GBIF_from_user_file.yml')

cfg = get_cfg_from_full_path(path_cfg)

'''
Home dir
    cfg['dir_home']
Where new images are stored
    cfg['dir_destination']
Where to read the occurance data
    cfg['filename_occ']
Where to read the image urls
    cfg['filename_img']
New file where combined records will be saved
    cfg['filename_combined']
'''
dir_home = cfg['dir_home']
dir_destination = cfg['dir_destination']
filename_occ = cfg['filename_occ']
filename_img = cfg['filename_img']
filename_combined = cfg['filename_combined']
filename_occ_appended = filename_occ
filename_img_appended = filename_img

n_already_downloaded = cfg['n_already_downloaded']
n_max_to_download = cfg['n_max_to_download']
n_imgs_per_species = cfg['n_imgs_per_species']
MP_low = cfg['MP_low']
MP_high = cfg['MP_high']

occ_new = pd.DataFrame()
img_new = pd.DataFrame()
validate_dir(dir_destination)

if cfg['do_use_wishlist'] == True:
    wishlist = cfg['wishlist']
    wishlist = pd.read_csv(os.path.join(dir_home,wishlist), header=0, low_memory=False)

    alreadyDownloaded, agg_occ, agg_img = download_subset_images_user_file(dir_home,dir_destination,n_already_downloaded,MP_low,MP_high,wishlist,filename_occ,filename_img)
    occ_new = occ_new.append(agg_occ,ignore_index = True)
    img_new = img_new.append(agg_img,ignore_index = True)

    occ_new.to_csv(os.path.join(dir_home,filename_occ_appended),index=False)
    img_new.to_csv(os.path.join(dir_home,filename_img_appended),index=False)
    # Merge both into one File?
    img_new = img_new.rename(columns={"identifier": "url"}) 
    combined = pd.concat([occ_new, img_new], axis=1, sort=False)
    combined.to_csv(os.path.join(dir_home,filename_combined),index=False)

    check_n_images_in_group(occ_new,n_imgs_per_species)
else:
    alreadyDownloaded, agg_occ, agg_img = download_subset_images_user_file_no_wishlist(dir_home,dir_destination,n_already_downloaded,MP_low,MP_high,filename_occ,filename_img)
    occ_new = occ_new.append(agg_occ,ignore_index = True)
    img_new = img_new.append(agg_img,ignore_index = True)

    occ_new.to_csv(os.path.join(dir_home,filename_occ_appended),index=False)
    img_new.to_csv(os.path.join(dir_home,filename_img_appended),index=False)
    # Merge both into one File?
    img_new = img_new.rename(columns={"identifier": "url"}) 
    combined = pd.concat([occ_new, img_new], axis=1, sort=False)
    combined.to_csv(os.path.join(dir_home,filename_combined),index=False)

    check_n_images_in_group(occ_new,n_imgs_per_species)