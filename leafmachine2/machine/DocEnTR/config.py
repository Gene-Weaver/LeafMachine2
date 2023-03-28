import argparse, os, inspect, sys, yaml

class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_path', type=str, default='//', help='specify your data path, better ending with the "/" ')
        self.parser.add_argument('--split_size', type=int, default=256, help= "better be a multiple of 8, like 128, 256, etc ..")
        self.parser.add_argument('--vit_patch_size', type=int, default=16 , help=" better be a multiple of 2 like 8, 16 etc ..")
        self.parser.add_argument('--vit_model_size', type=str, default='small', choices=['small', 'base', 'large'])
        self.parser.add_argument('--testing_dataset', type=str, default='RULER_test')
        self.parser.add_argument('--validation_dataset', type=str, default='RULER_val')
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--epochs', type=int, default=101, help= 'the desired training epochs')
        self.parser.add_argument('--model_weights_path', type=str, help= 'the desired trained model')
        
        
    def parse(self):
        return self.parser.parse_args()

class ConfigsDirect():
    data_path: str = ''
    split_size: str = ''
    vit_patch_size: str = ''
    vit_model_size: str = ''
    testing_dataset: str = ''
    validation_dataset: str = ''
    batch_size: str = ''
    epochs: str = ''
    model_weights_path: str = ''
    dir_save: str = ''
    gpu_id: int = 0

    def __init__(self):

        dir_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        path_cfg_private = os.path.join(dir_home,'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)
        if cfg_private['w_and_b']['w_and_b_key'] is not None:
            w_and_b_key = cfg_private['w_and_b']['w_and_b_key']

        path_to_config = dir_home
        cfg = load_cfg(path_to_config)

        if cfg['leafmachine']['ruler_DocEnTR_train']['data_path'] is None:
            self.data_path = ''
        else:
            self.data_path = cfg['leafmachine']['ruler_DocEnTR_train']['data_path']

        if cfg['leafmachine']['ruler_DocEnTR_train']['split_size'] is None:
            self.split_size = 256
        else:
            self.split_size = cfg['leafmachine']['ruler_DocEnTR_train']['split_size']

        if cfg['leafmachine']['ruler_DocEnTR_train']['vit_patch_size'] is None:
            self.vit_patch_size = 16
        else:
            self.vit_patch_size = cfg['leafmachine']['ruler_DocEnTR_train']['vit_patch_size']

        if cfg['leafmachine']['ruler_DocEnTR_train']['vit_model_size'] is None:
            self.vit_model_size = 'small'
        else:
            self.vit_model_size = cfg['leafmachine']['ruler_DocEnTR_train']['vit_model_size']

            
        if cfg['leafmachine']['ruler_DocEnTR_train']['testing_dataset'] is None:
            self.testing_dataset = ''
        else:
            self.testing_dataset = cfg['leafmachine']['ruler_DocEnTR_train']['testing_dataset']
            
        if cfg['leafmachine']['ruler_DocEnTR_train']['validation_dataset'] is None:
            self.validation_dataset = ''
        else:
            self.validation_dataset = cfg['leafmachine']['ruler_DocEnTR_train']['validation_dataset']

        if cfg['leafmachine']['ruler_DocEnTR_train']['batch_size'] is None:
            self.batch_size = 8
        else:
            self.batch_size = cfg['leafmachine']['ruler_DocEnTR_train']['batch_size']
        
        if cfg['leafmachine']['ruler_DocEnTR_train']['epochs'] is None:
            self.epochs = 101
        else:
            self.epochs = cfg['leafmachine']['ruler_DocEnTR_train']['epochs']

        if cfg['leafmachine']['ruler_DocEnTR_train']['model_weights_path'] is None:
            self.model_weights_path = ''
        else:
            self.model_weights_path = cfg['leafmachine']['ruler_DocEnTR_train']['model_weights_path']

        if cfg['leafmachine']['ruler_DocEnTR_train']['dir_save'] is None:
            self.dir_save = self.data_path
        else:
            self.dir_save = cfg['leafmachine']['ruler_DocEnTR_train']['dir_save']

        if cfg['leafmachine']['ruler_DocEnTR_train']['gpu_id'] is None:
            self.gpu_id = 0
        else:
            self.gpu_id = int(cfg['leafmachine']['ruler_DocEnTR_train']['gpu_id'])


def load_config_file(dir_home, cfg_file_path):
    if cfg_file_path == None: # Default path
        return load_cfg(dir_home)
    else:
        if cfg_file_path == 'test_installation':
            path_cfg = os.path.join(dir_home,'demo','LeafMachine2_demo.yaml')                     # TODO make the demo yaml
            return get_cfg_from_full_path(path_cfg)
        else: # Custom path
            return get_cfg_from_full_path(cfg_file_path)

def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg

def load_cfg(pathToCfg):
    try:
        with open(os.path.join(pathToCfg,"LeafMachine2.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except:
        with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)),"LeafMachine2.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    return cfg

