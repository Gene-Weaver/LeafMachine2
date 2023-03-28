# Test a checkpoint path during training
# OR
# Run the test after training

import os, argparse, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

from machine.general_utils import load_cfg, get_cfg_from_full_path

sys.path.insert(0, currentdir) 

from unicodedata import name
from evaluate_segmentation_to_pdf import evaluate_model_to_pdf

def main(cfg, dir_root):
    # DIR_MODEL = "leaf_seg__2022_09_23__13-07-23__POC_Dataset_10000_Iter_784PTS_CIOU_WK2__PR_mask_rcnn_R_50_FPN_3x"
    # DIR_TRAIN = os.path.abspath(os.path.join('detectron2','LM2_data','leaf_whole_part1','train','images'))
    # DIR_VAL = os.path.abspath(os.path.join('detectron2','LM2_data','leaf_whole_part1','val','images'))
    # DIR_CHECK = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'LM2_Env','Image_Datasets','GroundTruth_CroppedAnnotations_Group1_Partial','PLANT','Leaf_WHOLE')
    # cfg = load_cfg(args.path_to_cfg)
    evaluate_model_to_pdf(cfg, dir_root)

def parse_quick_test_to_pdf():
    parser = argparse.ArgumentParser(
            description='Parse inputs for quick_test_to_pdf.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--path-to-cfg',
                                type=str,
                                required=True,
                                help='')

    parser._action_groups.append(optional_args)
    args = parser.parse_args()


    return args

if __name__ == '__main__':
    dir_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path_cfg = os.path.join(dir_root,'LeafMachine2.yaml')
    cfg = get_cfg_from_full_path(path_cfg)
    # args = parse_quick_test_to_pdf()
    main(cfg, dir_root)
