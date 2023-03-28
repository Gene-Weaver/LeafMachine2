import os 
import pandas as pd

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

dir_file_wishlist = 'D:/Dropbox/Big_Grant/Sampling_plan.csv'
dir_all_families = 'D:\Dropbox\LM2_Env\Image_Datasets\GBIF_Ingest\GBIF_byFamily'

list_family_dirs = os.listdir(dir_all_families)
file_wishlist = pd.read_csv(dir_file_wishlist, header=0, low_memory=False, dtype=str)

uniq_family = pd.unique(file_wishlist['Family'])

common_families = intersection(uniq_family,list_family_dirs)

report = []
for fam in common_families:
    fam_occ = pd.read_csv(os.path.join(dir_all_families,fam,''.join([fam,'.csv'])), header=0, low_memory=False, dtype=str)
    file_wishlist_sub = file_wishlist[file_wishlist['Family'] == fam]
    print(f'FAMILY: {fam} SIZE: {file_wishlist_sub.shape[0]}')
    for index, row in file_wishlist_sub.iterrows():
        need_fam = row['Family']
        need_species = row['Taxon']

        keep = []
        try:
            if ',' in need_species:
                all_need_species = need_species.split(', ')
                
                for candidate in all_need_species:
                    if len(candidate) > 5:
                        keep.append(candidate)
            else:
                keep.append(need_species)
        except:
            keep = []

        for name in keep:
            name_good = True
            try:
                genus, species = name.split('_')
            except:
                try:
                    genus, species = name.split(' ')
                except:
                    name_good = False
            if name_good:
                fam_occ_specific = fam_occ[fam_occ['genus'] == genus]
                fam_occ_specific = fam_occ_specific[fam_occ_specific['specificEpithet'] == species]
                count = fam_occ_specific.shape[0]
                print(f'      {species} COUNT: {count}')
                report_row = {'Family': fam, 'Genus': genus, 'Species': species , 'Fullname': '_'.join([fam, genus, species]), 'Count': count}
                report.append(report_row)
report = pd.DataFrame(report)
report.to_csv('D:/Dropbox/Big_Grant/Sampling_plan_counts.csv',index=False,columns=['Family', 'Genus', 'Species', 'Fullname', 'Count'])