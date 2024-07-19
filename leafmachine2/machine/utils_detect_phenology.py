import os, glob, csv
from time import perf_counter

class PhenologyDetector:
    def __init__(self, cfg, base_dir):
        self.cfg = cfg
        self.base_dir = base_dir
        self.class_names = {
            0: 'leaf_whole',
            1: 'leaf_partial',
            2: 'leaflet',
            3: 'seed_fruit_one',
            4: 'seed_fruit_many',
            5: 'flower_one',
            6: 'flower_many',
            7: 'bud',
            8: 'specimen',
            9: 'roots',
            10: 'wood',
        }
        # Initialize counts as a dict of dicts
        self.file_counts = {}

    def count_annotations(self):
        MIN = self.cfg['leafmachine']['project']['minimum_total_reproductive_counts']

        label_files = glob.glob(os.path.join(self.base_dir, '*.txt'))
        for file_path in label_files:
            file_name = os.path.basename(file_path)
            # Initialize count for each class for this file
            self.file_counts[file_name] = {name: 0 for name in self.class_names.values()}
            with open(file_path, 'r') as file:
                for line in file:
                    class_index = int(line.split()[0])
                    class_name = self.class_names.get(class_index)
                    if class_name:
                        self.file_counts[file_name][class_name] += 1

            # Check for leaf presence (classes 0 and 1)
            # For ideal AND partial leaves
            if self.cfg['leafmachine']['project']['accept_only_ideal_leaves']:
                # For ONLY ideal leaves
                has_leaves = 1 if self.file_counts[file_name]['leaf_whole'] > 0 else 0
            else:
                has_leaves = 1 if self.file_counts[file_name]['leaf_whole'] > 0 or self.file_counts[file_name]['leaf_partial'] > 0 else 0

            # Check for fertility (classes 3, 4, 5, 6)
            
            is_fertile = 1 if any(self.file_counts[file_name][cls] > MIN for cls in ['seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many']) else 0

            # Update file-specific entry with has_leaves and is_fertile
            self.file_counts[file_name]['has_leaves'] = has_leaves
            self.file_counts[file_name]['is_fertile'] = is_fertile

    def get_counts(self):
        return self.file_counts

    def export_to_csv(self, output_path):
        fieldnames = ['file_name'] + list(self.class_names.values()) + ['has_leaves', 'is_fertile']
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file_name, counts in self.file_counts.items():
                row = {'file_name': file_name}
                row.update(counts)
                writer.writerow(row)

def detect_phenology(cfg, time_report, logger, dir_home, Project, Dirs):
    t2_start = perf_counter()
    logger.name = f'Detecting Phenology --- {Dirs.path_plant_components}'
    
    path_plant_labels = os.path.join(Dirs.path_plant_components, 'labels')
    
    detector = PhenologyDetector(cfg, path_plant_labels)
    detector.count_annotations()
    # counts = detector.get_counts()
    
    # Optionally, export the counts to a CSV file
    output_csv_path = os.path.join(Dirs.path_phenology, 'phenology.csv')
    detector.export_to_csv(output_csv_path)

    # Optionally, print the counts of each class for each file
    # for file_name, counts_per_file in counts.items():
    #     print(f"{file_name}:")
    #     for class_name, count in counts_per_file.items():
    #         print(f"  {class_name}: {count}")

    # return counts
    t2_stop = perf_counter()
    t_pheno = f"[Detecting Phenology elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_pheno)
    time_report['t_pheno'] = t_pheno
    return time_report