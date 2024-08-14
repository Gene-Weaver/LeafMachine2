# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.178'

import os
import sys
import inspect

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir1 = os.path.dirname(os.path.dirname(currentdir))
parentdir2 = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.append(currentdir)
sys.path.append(parentdir1)
sys.path.append(parentdir2)

# try:
#     from ultralytics.models import RTDETR, SAM, YOLO
#     from ultralytics.models.fastsam import FastSAM
#     from ultralytics.models.nas import NAS
#     from ultralytics.utils import SETTINGS as settings
#     from ultralytics.utils.checks import check_yolo as checks
#     from ultralytics.utils.downloads import download
# except:
from leafmachine2.keypoint_detector.ultralytics.models import RTDETR, SAM, YOLO
from leafmachine2.keypoint_detector.ultralytics.models.fastsam import FastSAM
from leafmachine2.keypoint_detector.ultralytics.models.nas import NAS
from leafmachine2.keypoint_detector.ultralytics.utils import SETTINGS as settings
from leafmachine2.keypoint_detector.ultralytics.utils.checks import check_yolo as checks
from leafmachine2.keypoint_detector.ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
