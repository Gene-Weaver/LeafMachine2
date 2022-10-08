import os
import datetime
from detectron2.config import get_cfg
from detectron2 import model_zoo


def leaf_config(ZOO_OPT,OUTPUT_DIR,DO_VAL,TRAIN_OR_DETECT,BATCH_SIZE,ITER,ITER_CK,ITER_WARM,N_WORKERS):
    cfg = get_cfg()

    # get configuration from model_zoo
    if ZOO_OPT == "mask_rcnn_R_50_FPN_3x": # yes
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
    elif ZOO_OPT == "mask_rcnn_R_50_FPN_1x":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    elif ZOO_OPT == "mask_rcnn_R_101_FPN_3x":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    elif ZOO_OPT == "mask_rcnn_X_101_32x8d_FPN_3x": # no
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Train and Val
    cfg.DATASETS.TRAIN = ("dataset_train",)
    if DO_VAL:
        cfg.DATASETS.TEST = ("dataset_val",)
    else:
        cfg.DATASETS.TEST = ()

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.TEST.EVAL_PERIOD = 500

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = N_WORKERS
    
    # Solver
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.WARMUP_ITERS = ITER_WARM
    cfg.SOLVER.MAX_ITER = ITER    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.CHECKPOINT_PERIOD = ITER_CK
    cfg.SOLVER.STEPS = []        # do not decay learning rate

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4],[8],[16],[32],[64]] # OR [[8],[16],[32],[64],[128]]

    # Input
    cfg.INPUT.FORMAT = "RGB"

    # day = "_".join([str(datetime.datetime.now().strftime("%Y")),str(datetime.datetime.now().strftime("%m")),str(datetime.datetime.now().strftime("%d"))])
    # time = "-".join([str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M")),str(datetime.datetime.now().strftime("%S"))])
    # new_time = "__".join([day,time])

    # cfg.OUTPUT_DIR = "__".join(["leaf_seg", new_time, NAME])
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    if TRAIN_OR_DETECT == "train":      
        try:  
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
            with open(os.path.join(cfg.OUTPUT_DIR,"cfg_output.yaml"), "w") as f:
                f.write(cfg.dump())   # save config to file
        except:
            print("Error: leaf_config.py")

    return cfg