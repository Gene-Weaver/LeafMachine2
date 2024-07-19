'''
venv requirements:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install cython
pip3 install opencv-python

git clone https://github.com/facebookresearch/detectron2.git

change:
IN: detectron2/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
//#ifdef WITH_CUDA ***MODIFIED***
#include "../box_iou_rotated/box_iou_rotated_utils.h"
//#endif ***MODIFIED***
// todo avoid this when pytorch supports "same directory" hipification
//#ifdef WITH_HIP ***MODIFIED***
#include "box_iou_rotated/box_iou_rotated_utils.h"
//#endif ***MODIFIED***

pip install -e .
'''
import os, json, datetime, yaml, wandb, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

from machine.general_utils import get_datetime, load_cfg, Print_Verbose_Error, get_cfg_from_full_path

sys.path.insert(0, currentdir) 

from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from datetime import timedelta
from dataclasses import dataclass, field

from leaf_config import leaf_config
from launch_leaf import launch
from evaluate_segmentation_to_pdf import evaluate_model_to_pdf
from leaf_config_pr import leaf_config_pr

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 
# parentdir = os.path.dirname(parentdir)
# sys.path.insert(0, parentdir) 

# from machine.general_utils import get_datetime

'''
For W&B
https://github.com/facebookresearch/detectron2/issues/774
'''

def get_dict(dir,json_name):
    f = open(os.path.join(dir,json_name))
    data = json.load(f)
    return data


def run(opts):  
    setup_logger()

    register_coco_instances("dataset_train", {}, opts.path_to_train_json, opts.dir_images_train)
    register_coco_instances("dataset_val", {}, opts.path_to_val_json, opts.dir_images_val)

    MetadataCatalog.get("dataset_train").set(thing_colors=[[0,255,46], [255,173,0],[255,0,209]])
    MetadataCatalog.get("dataset_val").set(thing_colors=[[0,255,46], [255,173,0],[255,0,209]])
    MetadataCatalog.get("dataset_train").set(thing_classes=['leaf','petiole','hole'])
    MetadataCatalog.get("dataset_val").set(thing_classes=['leaf','petiole','hole'])
    metadata = MetadataCatalog.get("dataset_val")

    # if "thing_colors" in metadata:
    #         pass
    # else:
    #     metadata['thing_colors'] = [[0,255,46], [255,173,0],[255,0,209]]]
    #     metadata['thing_classes'] = ['Leaf','Petiole','Hole']

    # Create config 
    cfg_leaf = leaf_config_pr(opts.model_name, 
                              opts.base_architecture,
                              opts.dir_out,
                              opts.do_validate_in_training,
                              "train",
                              opts.batch_size,
                              opts.iterations,
                              opts.checkpoint_freq,
                              opts.warmup,
                              opts.n_workers,
                              None)#opts.aug)
    print(cfg_leaf.dump())

    # Save metadata from the congig / dataset
    with open(os.path.join(cfg_leaf.OUTPUT_DIR,'metadata.json'), 'w') as outfile:
        json.dump(metadata.as_dict(), outfile)

    # Setup W & B
    wandb.login(key=opts.w_and_b_key)
    cfg_wandb = yaml.safe_load(cfg_leaf.dump())
    wandb.init(project=opts.project,name=opts.model_name, entity=opts.entity, config = cfg_wandb, sync_tensorboard=False)

    RESUME_TRAIN = True
    if RESUME_TRAIN:
        DIR_ROOT = os.getcwd()
        # cfg_leaf.OUTPUT_DIR = os.path.join(DIR_ROOT,'leaf_seg__2022_09_19__20-19-04')
        model_list = os.listdir(cfg_leaf.OUTPUT_DIR)
        if "model_final.pth" in model_list:
            model_to_use = os.path.join(cfg_leaf.OUTPUT_DIR, "model_final.pth")
        else:
            candidate = []
            for m in model_list:
                if "model" in m:
                    candidate.append(int(m.split("_")[1].split(".")[0]))
            model_to_use = [i for i, s in enumerate(model_list) if str(max(candidate)) in s][0]
            model_to_use = os.path.join(cfg_leaf.OUTPUT_DIR, model_list[model_to_use])
        cfg_leaf.MODEL.WEIGHTS = model_to_use
        cfg_leaf.SOLVER.BASE_LR = 0.00025
        cfg_leaf.SOLVER.WARMUP_ITERS = 0
        cfg_leaf.SOLVER.MAX_ITER = 200000
    
    trainer = DefaultTrainer(cfg_leaf) 
    if RESUME_TRAIN:
        trainer.resume_or_load(resume=True)
    else:
        trainer.resume_or_load(resume=False)
    
    wandb.watch(trainer.model, log_freq=20)
    trainer.train()

@dataclass
class TrainOptions:
    '''
    Default training values
        Increase or decrease batch_size, n_gpu, n_workers according to machine specs
    '''
    batch_size: int = 32
    iterations: int = 10000
    checkpoint_freq: int = 1000
    warmup: int = 1000
    n_gpus: int = 2
    n_workers: int = 8
    n_machines: int = 1
    default_timeout_minutes: int = 30
    base_architecture: str = 'PR_mask_rcnn_R_50_FPN_3x'
    do_validate_in_training: bool = True
    # aug: bool = False # not functional
    filename_train_json: str = 'POLYGONS_train.json'
    filename_val_json: str = 'POLYGONS_val.json'

    new_time: str = field(init=False)

    path_to_config: str = field(init=False)
    path_to_model: str = field(init=False)
    path_to_ruler_class_names: str = field(init=False)
    path_to_train_json: str = field(init=False)
    path_to_val_json: str = field(init=False)

    dir_images_train: str = field(init=False)
    dir_images_val: str = field(init=False)
    dir_images_test: str = field(init=False)
    dir_out: str = field(init=False)
    dir_root: str = field(init=False)

    model_name: str = field(init=False)
    
    cfg: str = field(init=False)
        
    w_and_b_key: str = field(init=False)
    project: str = field(init=False)
    entity: str = field(init=False)

    def __post_init__(self) -> None:
        '''
        Setup
        '''
        self.new_time = get_datetime()
        self.default_timeout_minutes = timedelta(self.default_timeout_minutes)
        '''
        Configure names
        '''
        self.dir_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        path_cfg_private = os.path.join(self.dir_root,'PRIVATE_DATA.yaml')
        self.cfg_private = get_cfg_from_full_path(path_cfg_private)

        self.path_to_config = self.dir_root
        self.cfg = load_cfg(self.path_to_config)
        if self.cfg['leafmachine']['segmentation_train']['model_options']['model_name'] is not None:
            self.model_name = self.cfg['leafmachine']['segmentation_train']['model_options']['model_name']
            self.model_name = "__".join([self.model_name,self.base_architecture])
            if self.cfg['leafmachine']['segmentation_train']['dir_out'] is None:
                self.dir_out = "__".join(["leaf_seg", self.new_time, self.model_name])
            else:
                self.dir_out = self.cfg['leafmachine']['segmentation_train']['dir_out']
        else:
            self.model_name = "DEFAULT_NAME"
            self.model_name = "__".join([self.model_name,self.base_architecture])
            if self.cfg['leafmachine']['segmentation_train']['dir_out'] is None:
                self.dir_out = "__".join(["leaf_seg", self.new_time, self.model_name])
                self.dir_out = os.path.join('models', self.dir_out)
            else:
                self.dir_out = self.cfg['leafmachine']['segmentation_train']['dir_out']
        '''
        Weights and Biases Info
        https://wandb.ai/site
        '''
        if self.cfg_private['w_and_b']['w_and_b_key'] is not None:
            self.w_and_b_key = self.cfg_private['w_and_b']['w_and_b_key']
        if self.cfg_private['w_and_b']['leaf_segmentation_project'] is not None:
            self.project = self.cfg_private['w_and_b']['leaf_segmentation_project']   
        if self.cfg_private['w_and_b']['entity'] is not None:
            self.entity = self.cfg_private['w_and_b']['entity']
        '''
        Setup dirs
        '''
        if self.cfg['leafmachine']['segmentation_train']['dir_images_train'] is not None:
            self.dir_images_train = self.cfg['leafmachine']['segmentation_train']['dir_images_train']
        else: 
            Print_Verbose_Error(self.cfg,1,'ERROR: Training directory is missing').print_error_to_console()
        if self.cfg['leafmachine']['segmentation_train']['dir_images_val'] is not None:
            self.dir_images_val = self.cfg['leafmachine']['segmentation_train']['dir_images_val']
        if self.cfg['leafmachine']['segmentation_train']['dir_images_test'] is not None:
            self.dir_images_test = self.cfg['leafmachine']['segmentation_train']['dir_images_test']
        '''
        Setup json files
        '''
        if self.cfg['leafmachine']['segmentation_train']['filename_train_json'] is not None:
            self.filename_train_json = self.cfg['leafmachine']['segmentation_train']['filename_train_json']
        if self.cfg['leafmachine']['segmentation_train']['filename_val_json'] is not None:
            self.filename_val_json = self.cfg['leafmachine']['segmentation_train']['filename_val_json']

        self.path_to_train_json = os.path.join(self.dir_images_train,self.filename_train_json)
        self.path_to_val_json = os.path.join(self.dir_images_val,self.filename_val_json)
        '''
        Model Name
        '''
        if self.cfg['leafmachine']['segmentation_train']['model_options']['model_name'] is not None:
            self.model_name = self.cfg['leafmachine']['segmentation_train']['model_options']['model_name']
        '''
        Model Training Options
        '''
        if self.cfg['leafmachine']['segmentation_train']['model_options']['base_architecture'] is not None:
            self.base_architecture = self.cfg['leafmachine']['segmentation_train']['model_options']['base_architecture']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['batch_size'] is not None:
            self.batch_size = self.cfg['leafmachine']['segmentation_train']['model_options']['batch_size']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['iterations'] is not None:
            self.iterations = self.cfg['leafmachine']['segmentation_train']['model_options']['iterations']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['checkpoint_freq'] is not None:
            self.checkpoint_freq = self.cfg['leafmachine']['segmentation_train']['model_options']['checkpoint_freq']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['warmup'] is not None:
            self.warmup = self.cfg['leafmachine']['segmentation_train']['model_options']['warmup']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['n_gpus'] is not None:
            self.n_gpus = self.cfg['leafmachine']['segmentation_train']['model_options']['n_gpus']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['n_machines'] is not None:
            self.n_machines = self.cfg['leafmachine']['segmentation_train']['model_options']['n_machines']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['n_workers'] is not None:
            self.n_workers = self.cfg['leafmachine']['segmentation_train']['model_options']['n_workers']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['default_timeout_minutes'] is not None:
            self.default_timeout_minutes = self.cfg['leafmachine']['segmentation_train']['model_options']['default_timeout_minutes']
        if self.cfg['leafmachine']['segmentation_train']['model_options']['do_validate_in_training'] is not None:
            self.do_validate_in_training = self.cfg['leafmachine']['segmentation_train']['model_options']['do_validate_in_training']
        # if self.cfg['leafmachine']['segmentation_train']['model_options']['apply_augmentation'] is not None:
            # self.aug = self.cfg['leafmachine']['segmentation_train']['model_options']['apply_augmentation']
        
if __name__ == '__main__':
    opts = TrainOptions()
    '''
    launch() allows multi-gpu training
    '''
    launch(
        run,
        opts.n_gpus,
        opts.n_machines,
        machine_rank=0,
        dist_url="auto",
        opts=opts,
        timeout=opts.default_timeout_minutes,
    )

    evaluate_model_to_pdf(opts.cfg, opts.dir_root)