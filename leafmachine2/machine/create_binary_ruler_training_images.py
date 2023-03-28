import os, cv2, random
import numpy as np

from general_utils import validate_dir

def create_training_images(dir_root, dir_out, random_keep_chance):
    for path, subdirs, files in os.walk(dir_root):
        if subdirs:
            # Dir containing dirs
            for subdir in subdirs:
                print(subdir)
                path_sub = os.path.join(dir_root,subdir)
                dir_out_sub = os.path.join(dir_out,subdir)
                validate_dir(dir_out_sub)

                files = os.listdir(path_sub)
                if len(files) > 50:
                    keep_list = random.sample(range(len(files)), 50)
                else:
                    keep_list = []

                for i, name in enumerate(files):
                    if len(files) <= 50:
                        do_process = True
                    else:
                        if i in keep_list:
                            do_process = True
                        else:
                            do_process = False

                    if do_process:
                        path_img = os.path.join(path_sub, name)
                        # print(os.path.join(subdir, name))
                        img = rotate_image(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2GRAY))
                        multi_threshold(img, dir_out_sub, name, random_keep_chance)

def multi_threshold(img, dir_out_sub, name, random_keep_chance):
    total = img.size
    for idx, i in enumerate(range(0, 255, 20)):
        name_bi = ''.join([name.split('.')[0], '_', str(i), '.jpg'])
        threshold_value = i
        img_bi = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]

        number_of_white = np.sum(img_bi == 255)
        number_of_black = np.sum(img_bi == 0)

        keep = do_keep(number_of_white, number_of_black, total, random_keep_chance)

        if keep:
            cv2.imwrite(os.path.join(dir_out_sub,name_bi), img_bi)

def do_keep(number_of_white, number_of_black, total, random_keep_chance):
    bounds_up = int(0.99*total)
    bounds_down = int(0.01*total)
    if (number_of_white == 0) or (number_of_black == 0):
        return random_keep(random_keep_chance)
    elif (number_of_white < bounds_down) or (number_of_black < bounds_down):
        return random_keep(random_keep_chance)
    elif (number_of_white > bounds_up) or (number_of_black > bounds_up):
        return random_keep(random_keep_chance)
    else:
        return True

def rotate_image(img):
    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def random_keep(random_keep_chance):
    return random.randrange(100) < random_keep_chance

if __name__ == '__main__':
    ### Create the training images
    dir_root = 'E:/TEMP_ruler/Rulers_ByType'
    dir_out = 'F:/binary_classifier_keep'
    random_keep_chance = 50 # 10 = 10percent

    create_training_images(dir_root, dir_out, random_keep_chance)

    # After manual QC, where I deleted all rejects, compare the keeps to the folder with all images,
    # delete the keeps from the folder with all images, leaving keep/reject
    
    # create_keep_and_reject_dirs()
