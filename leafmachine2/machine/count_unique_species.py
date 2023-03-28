import os
import csv

def get_species_list(directory):
    species = {}

    for filename in os.listdir(directory):
        species_name = filename.split('.')[0]
        species_name = species_name.split('_')[2:]
        species_name = '_'.join([species_name[0], species_name[1], species_name[2]])
        if species_name in species:
            species[species_name] += 1
        else:
            species[species_name] = 1

    species_list = [[key, value] for key, value in species.items()]

    with open('/home/brlab/Desktop/species_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Count'])
        writer.writerows(species_list)


if __name__ == '__main__':
    get_species_list('/media/brlab/e5827490-fff7-471f-a73d-e7ae3ea264bf/LM2_TEST/SET_Diospyros/images')