import os,json, cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from predictor_leaf import PredictorLeaf
from detectron2.projects import point_rend

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_dict(dir,json_name):
    f = open(os.path.join(dir,json_name))
    data = json.load(f)
    return data

class Detector_LM2: 
    def __init__(self,logger,DIR_MODEL,THRESH, LEAF_TYPE) -> None:
        # MODEL_ROOT = os.path.dirname(DIR_MODEL)

        self.cfg = get_cfg()
        try:
            self.cfg.merge_from_file(os.path.join(DIR_MODEL,'cfg_output.yaml'))
        except:
            point_rend.add_pointrend_config(self.cfg)
            self.cfg.merge_from_file(os.path.join(DIR_MODEL,'cfg_output.yaml'))

        
        # MetadataCatalog.get("dataset_val").set(thing_classes=['Leaf','Petiole','Hole'])
        # self.metadata = get_dict(self.cfg.OUTPUT_DIR,'metadata.json')
        self.metadata = get_dict(DIR_MODEL,'metadata.json')
        if "thing_colors" in self.metadata:
            try:
                if LEAF_TYPE == 0:
                    self.metadata['thing_colors'] = [[0,255,46], [255,173,0], [255, 0, 200]]
                    # self.metadata['thing_classes'] = ['leaf','petiole','hole']
                else:
                    self.metadata['thing_colors'] = [[255,200,0], [0, 140, 255], [255, 0, 200]]
                    # self.metadata['thing_classes'] = ['leaf','petiole','hole']
            except:
                pass
        else:
            if LEAF_TYPE == 0:
                self.metadata['thing_colors'] = [[0,255,46], [255,173,0], [255, 0, 200]]
            else:
                self.metadata['thing_colors'] = [[255,200,0], [0, 140, 255], [255, 0, 200]]
            self.metadata['thing_classes'] = ['leaf','petiole','hole']
            # with open(os.path.join(self.cfg.OUTPUT_DIR,'metadata.json'), 'w') as outfile:
            with open(os.path.join(DIR_MODEL,'metadata.json'), 'w') as outfile:
                json.dump(self.metadata, outfile)
        # DatasetCatalog.register(self.metadata)
        # MetadataCatalog.get("dataset_val").set(thing_colors=[[0,0,0],[0,255,46], [255,173,0]],thing_classes=['Leaf','Petiole','Hole'])

        model_list = os.listdir(DIR_MODEL)
        if "model_final.pth" in model_list:
            model_to_use = os.path.join(DIR_MODEL, "model_final.pth")
            self.model_to_use_name = "model_final.pth"
            # print(f'{bcolors.OKCYAN}Using Final Model: "model_final.pth"{bcolors.ENDC}')
            logger.info(f'Using Final Model: {model_to_use}')

        else:
            candidate = []
            for m in model_list:
                if '.pdf' not in m:
                    if "model" in m:
                        candidate.append(int(m.split("_")[1].split(".")[0]))
            model_to_use_name = [i for i, s in enumerate(model_list) if str(max(candidate)) in s][0]
            self.model_to_use_name = model_list[model_to_use_name]
            model_to_use = os.path.join(DIR_MODEL, self.model_to_use_name)
            # print(f'{bcolors.WARNING}Using Checkpoint Model: {self.model_to_use_name}{bcolors.ENDC}')
            logger.info(f'Using Checkpoint Model: {self.model_to_use_name}')


        self.cfg.MODEL.WEIGHTS = model_to_use # "model_final.pth"  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH
        
        self.predictor = PredictorLeaf(self.cfg)

    def onImage(self, image_path,SHOW_IMG):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictions = self.predictor(img)

        v = Visualizer(img[:, :, ::-1],
                metadata=self.metadata,
                scale=1, 
                instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        plt.ioff()
        fig = plt.figure(figsize=(10,10),dpi=100, facecolor='black')
        plt.imshow(out.get_image()[:,:,::-1])

        if SHOW_IMG:
            plt.ion()
            plt.show()
        
        plt.close()
        return fig
    
    def segment(self, img, generate_overlay, overlay_dpi, bg_color):
        # img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictions = self.predictor(img)

        v = Visualizer(img[:, :, ::-1],
                metadata=self.metadata,
                scale=1, 
                instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out, out_polygons, out_bboxes, out_labels, out_color = v.draw_instance_predictions_LM2(predictions["instances"].to("cpu"))
        
        # DISABLED, was not using the built-in fig generator
        # if generate_overlay:
        #     plt.ioff()
        #     fig = plt.figure(figsize=(out.width/100,out.height/100),dpi=overlay_dpi, facecolor=bg_color)
        #     plt.imshow(out.get_image()[:,:,::-1])

        #     # fig.savefig("figure.png")

        #     plt.close()
        # else:
        #     fig = None

        # return fig, out_polygons, out_bboxes, out_labels, out_color
        return out_polygons, out_bboxes, out_labels, out_color

class Detector: 
    def __init__(self,DIR_MODEL,THRESH) -> None:
        # MODEL_ROOT = os.path.dirname(DIR_MODEL)

        self.cfg = get_cfg()
        try:
            self.cfg.merge_from_file(os.path.join(DIR_MODEL,'cfg_output.yaml'))
        except:
            point_rend.add_pointrend_config(self.cfg)
            self.cfg.merge_from_file(os.path.join(DIR_MODEL,'cfg_output.yaml'))

        
        # MetadataCatalog.get("dataset_val").set(thing_classes=['Leaf','Petiole','Hole'])
        # self.metadata = get_dict(self.cfg.OUTPUT_DIR,'metadata.json')
        self.metadata = get_dict(DIR_MODEL,'metadata.json')
        if "thing_colors" in  self.metadata:
            pass
        else:
            self.metadata['thing_colors'] = [[0,0,0],[0,255,46], [255,173,0]]
            self.metadata['thing_classes'] = ['Leaf','Petiole','Hole']
            # with open(os.path.join(self.cfg.OUTPUT_DIR,'metadata.json'), 'w') as outfile:
            with open(os.path.join(DIR_MODEL,'metadata.json'), 'w') as outfile:
                json.dump(self.metadata, outfile)
        # DatasetCatalog.register(self.metadata)
        # MetadataCatalog.get("dataset_val").set(thing_colors=[[0,0,0],[0,255,46], [255,173,0]],thing_classes=['Leaf','Petiole','Hole'])

        model_list = os.listdir(DIR_MODEL)
        if "model_final.pth" in model_list:
            model_to_use = os.path.join(DIR_MODEL, "model_final.pth")
            self.model_to_use_name = "model_final.pth"
            print(f'{bcolors.OKCYAN}Using Final Model: "model_final.pth"{bcolors.ENDC}')
        else:
            candidate = []
            for m in model_list:
                if '.pdf' not in m:
                    if "model" in m:
                        candidate.append(int(m.split("_")[1].split(".")[0]))
            model_to_use_name = [i for i, s in enumerate(model_list) if str(max(candidate)) in s][0]
            self.model_to_use_name = model_list[model_to_use_name]
            model_to_use = os.path.join(DIR_MODEL, self.model_to_use_name)
            print(f'{bcolors.WARNING}Using Checkpoint Model: {self.model_to_use_name}{bcolors.ENDC}')

        self.cfg.MODEL.WEIGHTS = model_to_use # "model_final.pth"  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESH
        
        self.predictor = PredictorLeaf(self.cfg)

    def onImage(self, image_path,SHOW_IMG):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictions = self.predictor(img)

        v = Visualizer(img[:, :, ::-1],
                metadata=self.metadata,
                scale=1, 
                instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        plt.ioff()
        fig = plt.figure(figsize=(10,10),dpi=100, facecolor='black')
        plt.imshow(out.get_image()[:,:,::-1])

        if SHOW_IMG:
            plt.ion()
            plt.show()
        
        plt.close()
        return fig
    
    def segment(self, img, generate_overlay, overlay_dpi, bg_color):
        # img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictions = self.predictor(img)
        metadata = self.metadata

        v = Visualizer(img[:, :, ::-1],
                metadata=self.metadata,
                scale=1, 
                instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out, out_polygons, out_bboxes, out_labels, out_color = v.draw_instance_predictions_LM2(predictions["instances"].to("cpu"))
        
        if generate_overlay:
            plt.ioff()
            fig = plt.figure(figsize=(out.width/100,out.height/100),dpi=overlay_dpi, facecolor=bg_color)
            plt.imshow(out.get_image()[:,:,::-1])

            # fig.savefig("figure.png")

            plt.close()
        else:
            fig = None

        return fig, out_polygons, out_bboxes, out_labels, out_color