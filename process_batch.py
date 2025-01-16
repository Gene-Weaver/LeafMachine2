import os, yaml

from leafmachine2.machine.general_utils import load_config_file
from leafmachine2.machine.machine import machine
from leafmachine2.machine.email_updates import send_update

def save_config_file(cfg, file_path):
    """Save the modified configuration back to a file."""
    with open(file_path, 'w') as file:
        yaml.dump(cfg, file)
    print(f"Configuration saved to {file_path}")

def prep(dir_to_process):
    dir_home = os.path.dirname(__file__)
    cfg_file_path = None
    cfg_testing = None

    # Get a list of full paths to each family folder in the directory
    family_paths = [
        os.path.join(dir_to_process, family)
        for family in os.listdir(dir_to_process)
        if os.path.isdir(os.path.join(dir_to_process, family))
    ]

    # Check each family folder for the "LM2" subfolder
    for family_path in family_paths:
        lm2_folder = os.path.join(family_path, "LM2")
        img_folder = os.path.join(family_path, "img")
        if os.path.exists(lm2_folder) and os.path.isdir(lm2_folder):
            # Skip if "LM2" folder already exists
            print(f"Skipping: {family_path} - 'LM2' folder already exists")
        else:
            if not os.path.exists(img_folder):
                print(f"Skipping: {family_path} - 'img' folder does not exist")
            elif os.path.exists(img_folder) and not os.listdir(img_folder):
                print(f"Skipping: {family_path} - 'img' folder exists but is empty")
            else:
                # Print message if "LM2" folder is missing
                print(f"Needs to be processed with LM2: {family_path}")

                cfg = load_config_file(dir_home, cfg_file_path, system='LeafMachine2')

                cfg['leafmachine']['project']['dir_images_local'] = os.path.join(family_path, 'img')
                cfg['leafmachine']['project']['dir_output'] = family_path
                cfg['leafmachine']['project']['run_name'] = "LM2"
                cfg['leafmachine']['project']['batch_size'] = 20000
                cfg['leafmachine']['project']['num_gpus'] = 2
                cfg['leafmachine']['project']['num_workers'] = 32
                cfg['leafmachine']['project']['num_workers_cropping'] = 32
                cfg['leafmachine']['project']['num_workers_overlay'] = 32
                cfg['leafmachine']['project']['num_workers_ruler'] = 32
                cfg['leafmachine']['project']['num_workers_seg'] = 24

                try:
                    output_cfg_path = os.path.join(dir_home, "LeafMachine2.yaml")  # Save the modified config per family
                    save_config_file(cfg, output_cfg_path)
                except Exception as e:
                    print(f"Error saving config to {output_cfg_path}: {e}")


                try:
                    send_update(family_path, "LM2 Starting ---", pc="ada")
                    machine(cfg_file_path, dir_home, cfg_testing, progress_report=None)
                    send_update(family_path, "LM2 Complete! ---", pc="ada")

                except Exception as e:
                    send_update(family_path, "LM2 Failed ---", pc="ada")
                    print(f"Error saving config to {output_cfg_path}: {e}")
                
                


if __name__ == "__main__":

    dir_to_process = "/media/nas/GBIF_Downloads/Magnoliopsida_By_Family"

    prep(dir_to_process)

