'''
Intake a directory, or dir containing other dirs, and squarify all rulers inside
'''

import os, cv2, argparse
from ruler_utils import validate_dir, squarify, squarify_quartiles, squarify_nine, squarify_tile_four_versions

def squarify_imgs_in_dir(dirToSquare,dirOut,makeSquare,sz):
    validate_dir(dirOut)
    doFiles = True
    for path, subdirs, files in os.walk(dirToSquare):
        if subdirs:
            doFiles = False
            # Dir containing dirs
            for subdir in subdirs:
                print(subdir)
                subPath = os.path.join(dirToSquare,subdir)
                dirOutSub = os.path.abspath(os.path.join(dirOut,subdir))
                validate_dir(dirOutSub)

                files = os.listdir(subPath)
                for name in files:
                    imgPath=os.path.join(subPath, name)
                    # print(os.path.join(subdir, name))
                    img = cv2.imread(imgPath)
                    img = squarify_tile_four_versions(img,False,makeSquare,sz)
                    cv2.imwrite(os.path.join(dirOutSub,name),img)
        else: # Dir containing images only
            if doFiles:
                for name in files:
                    imgPath=os.path.join(path, name)
                    # print(os.path.join(subdir, name))
                    img = cv2.imread(imgPath)
                    img = squarify_tile_four_versions(img,False,makeSquare,sz)
                    cv2.imwrite(os.path.join(dirOut,name),img)
            else:
                break

# def main(dirToSquare,dirOut,makeSquare,sz):
#     dirToSquare = os.path.join('E:','TEMP_ruler','Rulers_ByType')
#     dirOut =  os.path.join('E:','TEMP_ruler','Rulers_ByType_Squarify')
#     makeSquare = True
#     sz = 360
#     validate_dir(dirOut)
#     squarify_imgs_in_dir(dirToSquare,dirOut,makeSquare,sz)

def parse_squarify():
    parser = argparse.ArgumentParser(
            description='Parse inputs for squarify_imgs_in_dir.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--dir-images',
                                type=str,
                                required=True,
                                help='')
    required_args.add_argument('--dir-out',
                                type=str,
                                required=True,
                                help='')
    required_args.add_argument('--make-square',
                                type=bool,
                                required=True,
                                help='print results: class and certainty')
    required_args.add_argument('--size',
                                type=int,
                                required=True,
                                help='')
    parser._action_groups.append(optional_args)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_squarify()
    squarify_imgs_in_dir(args.dir_images,args.dir_out,args.make_square,args.size)