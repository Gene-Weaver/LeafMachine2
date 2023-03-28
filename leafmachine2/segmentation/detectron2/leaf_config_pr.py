import os
import datetime
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.projects import point_rend

def leaf_config_pr(MODEL_NAME, ZOO_OPT,OUTPUT_DIR,DO_VAL,TRAIN_OR_DETECT,BATCH_SIZE,ITER,ITER_CK,ITER_WARM,N_WORKERS,AUG):
    cfg = get_cfg()

    '''
    CUDA 11.3 - export PATH=/usr/local/cuda-11.3/bin:$PATH

    https://colab.research.google.com/drive/1isGPL5h5_cKoPPhVL9XhMokRtHDvmMVL#scrollTo=HUjkwRsOn1O0

    # clone the repo in order to access pre-defined configs in PointRend project
    !git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
    # install detectron2 from source
    !pip install -e detectron2_repo
    # See https://detectron2.readthedocs.io/tutorials/install.html for other installation options
    # '''


    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    dir_PR = os.path.dirname(__file__)
    cfg.merge_from_file(os.path.join(dir_PR,'projects','PointRend','configs','InstanceSegmentation','pointrend_rcnn_R_50_FPN_3x_coco.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(dir_PR,'models','PointRend','point_rend_baseline.pkl')
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)

    # get configuration from model_zoo
    # if ZOO_OPT == "mask_rcnn_R_50_FPN_3x": # yes
    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 


    # Train and Val
    cfg.DATASETS.TRAIN = ("dataset_train",)
    if DO_VAL:
        cfg.DATASETS.TEST = ("dataset_val",)
    else:
        cfg.DATASETS.TEST = ()

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 10
    cfg.TEST.EVAL_PERIOD = 500

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = N_WORKERS
    
    # Solver
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR 0.00025 
    cfg.SOLVER.WARMUP_ITERS = ITER_WARM
    cfg.SOLVER.MAX_ITER = ITER    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.CHECKPOINT_PERIOD = ITER_CK
    cfg.SOLVER.STEPS = []        # do not decay learning rate

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 3 
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1" # default "smooth_l1" Options are: "smooth_l1", "giou", "diou", "ciou"

    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 1176 # 392 # default = 196 = 14*14
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 42*42# 28 * 28

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4],[8],[16],[32],[64]] # OR [[8],[16],[32],[64],[128]]

    # Input
    cfg.INPUT.FORMAT = "BGR"

    # if AUG:
    #     cfg.AUG = 'LM2'
    # else:
    #     cfg.AUG = None

    # day = "_".join([str(datetime.datetime.now().strftime("%Y")),str(datetime.datetime.now().strftime("%m")),str(datetime.datetime.now().strftime("%d"))])
    # time = "-".join([str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M")),str(datetime.datetime.now().strftime("%S"))])
    # new_time = "__".join([day,time])

    # cfg.OUTPUT_DIR = "__".join(["leaf_seg", new_time, MODEL_NAME])
    # cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, cfg.OUTPUT_DIR)
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME)
    # cfg.OUTPUT_DIR = OUTPUT_DIR
    
    if TRAIN_OR_DETECT == "train":      
        try:  
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
            with open(os.path.join(cfg.OUTPUT_DIR,"cfg_output.yaml"), "w") as f:
                f.write(cfg.dump())   # save config to file
        except:
            print("Error: leaf_config_pr.py")

    return cfg