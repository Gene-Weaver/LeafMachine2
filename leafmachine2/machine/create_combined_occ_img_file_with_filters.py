import os, time, requests, yaml, re, csv, sys, inspect
import pandas as pd
import numpy as np
from tqdm import tqdm

from general_utils import bcolors, validate_dir

# Return entire row of file_to_search that matches the gbif_id, else return []
def find_gbifID(gbif_id,file_to_search):
    row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)),:]
    if row_found.empty:
        # print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
        row_found = None
    else:
        # print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
        pass
    return row_found

def validate_herb_code(occ_row):
    # print(occ_row)
    # Herbarium codes are not always in the correct column, we need to find the right one
    try:
        opts = [occ_row['institutionCode'],
            occ_row['institutionID'],
            occ_row['ownerInstitutionCode'],
            occ_row['collectionCode'],
            occ_row['publisher'],
            occ_row['occurrenceID']]
        opts = [item for item in opts if not(pd.isnull(item.values)) == True]
    except:
        opts = [str(occ_row['institutionCode']),
            str(occ_row['institutionID']),
            str(occ_row['ownerInstitutionCode']),
            str(occ_row['collectionCode']),
            str(occ_row['publisher']),
            str(occ_row['occurrenceID'])]
        opts = pd.DataFrame(opts)
        opts = opts.dropna()
        opts = opts.apply(lambda x: x[0]).tolist()

    opts_short = []

    for word in opts:
        #print(word)
        if len(word) <= 8:
            if word is not None:
                opts_short = opts_short + [word]

    if len(opts_short) == 0:
        try:
            herb_code = occ_row['publisher'].values[0].replace(" ","-")
        except:
            try:
                herb_code = occ_row['publisher'].replace(" ","-")
            except:
                herb_code = "ERROR"
    try:
        inst_ID = occ_row['institutionID'].values[0]
        occ_ID = occ_row['occurrenceID'].values[0]
    except:
        inst_ID = occ_row['institutionID']
        occ_ID = occ_row['occurrenceID']
    if inst_ID == "UBC Herbarium":
        herb_code = "UBC"
    elif inst_ID == "Naturalis Biodiversity Center":
        herb_code = "L"
    elif inst_ID == "Forest Herbarium Ibadan (FHI)":
        herb_code = "FHI"
    elif 'id.luomus.fi' in occ_ID:
        herb_code = "FinBIF"
    else:
        if len(opts_short) > 0:
            herb_code = opts_short[0]

    try:
        herb_code = herb_code.values[0]
    except:
        herb_code = herb_code

    # Specific cases that require manual overrides
    # If you see an herbarium DWC file with a similar error, add them here
    if herb_code == "Qarshi-Botanical-Garden,-Qarshi-Industries-Pvt.-Ltd,-Pakistan":
        herb_code = "Qarshi-Botanical-Garden"
    elif herb_code == "12650":
        herb_code = "SDSU"
    elif herb_code == "322":
        herb_code = "SDSU"
    elif herb_code == "GC-University,-Lahore":
        herb_code = "GC-University-Lahore"
    elif herb_code == "Institute-of-Biology-of-Komi-Scientific-Centre-of-the-Ural-Branch-of-the-Russian-Academy-of-Sciences":
        herb_code = "Komi-Scientific-Centre"
    
    return herb_code

def remove_illegal_chars(text):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]","",text)
    return cleaned

def keep_first_word(text):
    if (' ' in text) == True:
        cleaned = text.split(' ')[0]
    else:
        cleaned = text
    return cleaned

# Create a filename for the downloaded image
# In the case sensitive format:
#        HERBARIUM_barcode_Family_Genus_species.jpg
def generate_image_filename(occ_row):
    herb_code = remove_illegal_chars(validate_herb_code(occ_row))
    try:
        specimen_id = str(occ_row['gbifID'].values[0])
        family = remove_illegal_chars(occ_row['family'].values[0])
        genus = remove_illegal_chars(occ_row['genus'].values[0])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet'].values[0]))
    except:
        specimen_id = str(occ_row['gbifID'])
        family = remove_illegal_chars(occ_row['family'])
        genus = remove_illegal_chars(occ_row['genus'])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet']))
    fullname = '_'.join([family, genus, species])

    filename_image = '_'.join([herb_code, specimen_id, fullname])
    filename_image_jpg = '.'.join([filename_image, 'jpg'])

    return filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname



def filter_occ_img_files(cfg):
    dir_home = cfg['dir_home']
    dir_destination_images = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']
    filename_occ = cfg['filename_occ']
    filename_img = cfg['filename_img']
    filename_combined = cfg['filename_combined']
    n_imgs_per_species = cfg['n_imgs_per_species']
    n_imgs_per_genus = cfg['n_imgs_per_genus']
    n_imgs_per_family = cfg['n_imgs_per_family']
    n_imgs_total = cfg['n_imgs_total']
    do_shuffle_occ = cfg['do_shuffle_occ']

    # Initialize counters
    family_counter = {}
    genus_counter = {}
    fullname_counter = {}
    total_counter = 0

    # Initialize empty lists to hold aggregated DataFrames
    agg_occ = []
    agg_img = []

    occ_all = pd.read_table(os.path.join(dir_home, filename_occ), header=0, low_memory=False)
    img_all = pd.read_table(os.path.join(dir_home, filename_img), header=0, low_memory=False)
    
    # Filter based on 'gbifID'
    gbif_ids = set(occ_all['gbifID']).intersection(set(img_all['gbifID']))
    occ_all_filtered = occ_all[occ_all['gbifID'].isin(gbif_ids)]
    img_all_filtered = img_all[img_all['gbifID'].isin(gbif_ids)]

    # shuffle
    if do_shuffle_occ:
        gbif_ids_series = pd.Series(list(gbif_ids))
        gbif_ids_series = gbif_ids_series.sample(frac=1, random_state=2023).reset_index(drop=True)
        gbif_ids_list = gbif_ids_series.tolist()    

    for gbif_id in tqdm(gbif_ids_list, desc='Processing', unit='rows'):
        occ_row = find_gbifID(gbif_id, occ_all_filtered)
        img_row = find_gbifID(gbif_id, img_all_filtered)

        # Skip to the next iteration if no matching row is found
        if occ_row is None or img_row is None:
            continue

        try:
            filename_image, _, _, specimen_id, family, genus, species, fullname = generate_image_filename(occ_row)
            
            # Apply filters based on counts and cfg values
            if family not in family_counter:
                family_counter[family] = 0
            if genus not in genus_counter:
                genus_counter[genus] = 0
            if fullname not in fullname_counter: # this is for species
                fullname_counter[fullname] = 0
            
            # Check each condition, treating None as "no limit"
            check_family = family_counter[family] < n_imgs_per_family if n_imgs_per_family is not None else True
            check_genus = genus_counter[genus] < n_imgs_per_genus if n_imgs_per_genus is not None else True
            check_species = fullname_counter[fullname] < n_imgs_per_species if n_imgs_per_species is not None else True
            check_total = total_counter < n_imgs_total if n_imgs_total is not None else True

            if occ_row.shape[0] == 1:
                if img_row.shape[0] == 3:
                    img_row = img_row.iloc[1:2].reset_index(drop=True)  # Take the middle row and reset the index
                    do_keep_row = True
                elif img_row.shape[0] == 1:
                    img_row = img_row.reset_index(drop=True)  # Reset the index
                    do_keep_row = True
                else:
                    do_keep_row = False

                if do_keep_row:
                    if check_family and check_genus and check_species and check_total:
                        family_counter[family] = family_counter.get(family, 0) + 1
                        genus_counter[genus] = genus_counter.get(genus, 0) + 1
                        fullname_counter[fullname] = fullname_counter.get(fullname, 0) + 1
                        total_counter += 1

                        occ_row = occ_row.copy()
                        occ_row = occ_row.assign(fullname=fullname)  # Safer way to add a column

                        img_row = img_row.copy()
                        img_row = img_row.assign(fullname=fullname)

                        agg_occ.append(occ_row)
                        agg_img.append(img_row)
            else:
                pass

        except Exception as e:
            tqdm.write(f"Error processing gbifID {gbif_id}: {e}")
            continue

    # Use ignore_index=True to reset index when concatenating
    agg_occ = pd.concat(agg_occ, ignore_index=True) if agg_occ else pd.DataFrame()
    agg_img = pd.concat(agg_img, ignore_index=True) if agg_img else pd.DataFrame()

    return None, agg_occ, agg_img


def check_n_images_in_group(detailedOcc,N):
    fam = detailedOcc['fullname'].unique()
    for f in fam:
        ct = len(detailedOcc[detailedOcc['fullname'].str.match(f)])
        if ct == N:
            print(f"{bcolors.OKGREEN}{f}: {ct}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}{f}: {ct}{bcolors.ENDC}")

def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg


if __name__ == "__main__":
    dir_LM2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_current_config = os.path.join(dir_LM2,'configs')
    path_cfg = os.path.join(dir_current_config,'config_create_occ_img_file_with_filters.yaml')

    cfg = get_cfg_from_full_path(path_cfg)

    dir_home = cfg['dir_home']
    dir_destination_images = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']
    filename_occ_appended = cfg['filename_occ_appended']
    filename_img_appended = cfg['filename_img_appended']
    filename_combined = cfg['filename_combined']
    n_imgs_per_species = cfg['n_imgs_per_species']

    validate_dir(dir_destination_csv)
    validate_dir(dir_destination_images)

    alreadyDownloaded, agg_occ, agg_img = filter_occ_img_files(cfg)

    # Initialize occ_new and img_new or read from existing CSVs
    occ_new = pd.DataFrame() # Initialize as empty DataFrame or read from an existing CSV
    img_new = pd.DataFrame() # Initialize as empty DataFrame or read from an existing CSV

    occ_new = pd.concat([occ_new, agg_occ], ignore_index=True)
    img_new = pd.concat([img_new, agg_img], ignore_index=True)

    occ_new.to_csv(os.path.join(dir_destination_csv, filename_occ_appended), index=False)
    img_new.to_csv(os.path.join(dir_destination_csv, filename_img_appended), index=False)

    img_new = img_new.rename(columns={"identifier": "url"})
    combined = pd.concat([occ_new, img_new], axis=1, sort=False)

    combined.to_csv(os.path.join(dir_destination_csv, filename_combined), index=False)

    # check_n_images_in_group(occ_new, n_imgs_per_species)