# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
import os
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics import YOLO

class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml', epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'pose'
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = PoseModel(cfg, ch=3, nc=self.data['nc'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data['kpt_shape']

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return yolo.pose.PoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch['img']
        kpts = batch['keypoints']
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        plot_images(images,
                    batch_idx,
                    cls,
                    bboxes,
                    kpts=kpts,
                    paths=paths,
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8x-pose.pt')  # load a pretrained model (recommended for training)
    # model = YOLO(os.path.join(os.getcwd(),'leafmachine2','keypoint_detector','ultralytics','models','yolo','pose', 'trained_models','RESUME_uniform_spaced_640_ptFrom__best_640_uniform_spaced_e150.pt'))  # load a pretrained model (recommended for training)
    # model = YOLO(os.path.join(os.getcwd(),'KP_2024','uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2_kobj40','weights','last.pt'))  # load a pretrained model (recommended for training)
    print(os.getcwd())
    cfg = os.path.join(os.getcwd(),'leafmachine2','keypoint_detector','ultralytics','models','yolo','pose', 'corner_skeleton.yaml')
    print(cfg)
    # Train the model
    # https://docs.ultralytics.com/usage/cfg/#modes
    results = model.train(data=cfg, epochs=1000, imgsz=640,
                        device=[0, 1],
                        batch=64, #32
                        cache=True,
                        workers=32,
                        project = 'Herbarium_Sheet_Corners',
                        name = 'corner_detector',
                        pretrained=True,
                        resume=False,
                        close_mosaic=500,
                        pose=20.0, # best = 20.0
                        box=2.0,
                        kobj=40.0, # best = 20.0
                        patience=0,
                        single_cls=True, # new
                        label_smoothing=0.5, # best = 0.5 # between 0.0 and 1.0
                        plots=True,# new

                        # lr0=0.002,
                        warmup_epochs=50,
                        )#, device=[0, 1])

