from leafmachine2.machine.machine import machine

if __name__ == '__main__':
    # To use LeafMachine2.yaml file, set cfg_file_path = None
    # To switch between config files, you can provide the full path to a different config file
    cfg_file_path = None
    machine(cfg_file_path)