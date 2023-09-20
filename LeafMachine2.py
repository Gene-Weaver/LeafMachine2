import os
from leafmachine2.machine.machine import machine

if __name__ == '__main__':
    # To use LeafMachine2.yaml file, set cfg_file_path = None
    # To switch between config files, you can provide the full path to a different config file
    cfg_file_path = None

    # Set LeafMachine2 dir 
    dir_home = os.path.dirname(__file__)

    machine(cfg_file_path, dir_home, cfg_test=None)
