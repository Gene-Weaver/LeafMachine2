from leafmachine2.machine.machine_detect_phenology import machine
import os
if __name__ == '__main__':
    # To use LeafMachine2.yaml file, set cfg_file_path = None
    # To switch between config files, you can provide the full path to a different config file
    cfg_file_path = None
    cfg_testing = None
    dir_home = os.path.dirname(__file__)
    print(dir_home)
    machine(cfg_file_path, dir_home, cfg_testing)
