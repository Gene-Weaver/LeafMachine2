import os, cv2, random, shutil

def delete_matching_files(keep_dir, reject_dir):
    """
    Deletes matching files in reject_dir based on files in keep_dir.
    """
    for subdir, dirs, files in os.walk(keep_dir):
        for file in files:
            keep_file_path = os.path.join(subdir, file)
            reject_file_path = keep_file_path.replace(keep_dir, reject_dir)
            if os.path.exists(reject_file_path):
                os.remove(reject_file_path)
                print(f"Deleted {reject_file_path}")


def copy_all_images_to_new_dir(src_dir, dest_dir):
    """
    Copies all image files from subdirectories of src_dir to a new directory dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if file is an image file
            if file.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                # Construct new file path
                src_file_path = os.path.join(subdir, file)
                dest_file_path = os.path.join(dest_dir, file)

                # Copy file to new directory
                shutil.copy(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")

def match_DocEnTR_binary_with_parent_rgb(src_dir, all_imgs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Ruler__ANHC_3435785516_Altingiaceae_Liquidambar_styraciflua__1_140.jpg
    for file in os.listdir(src_dir):
        # Check if file is an image file
        if file.endswith((".jpg")):
            parent_file = file.rsplit('_', 1)[0]

            png_name = '.'.join([file.split('.')[0], 'png'])

            img_jpg = cv2.imread(os.path.join(all_imgs, '.'.join([parent_file, 'jpg'])))
            cv2.imwrite(os.path.join(dest_dir, png_name), img_jpg)

            img_bi = cv2.imread(os.path.join(src_dir, file))
            cv2.imwrite(os.path.join(src_dir, png_name), img_bi)

            os.remove(os.path.join(src_dir, file))


def split_train_val_test(imgs_dir, gt_imgs_dir, output_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    # create directories for train, val, test splits
    train_dir = os.path.join(output_dir, 'train/')
    val_dir = os.path.join(output_dir, 'val/')
    test_dir = os.path.join(output_dir, 'test/')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # create directories for imgs and gt_imgs
    train_imgs_dir = os.path.join(train_dir, 'imgs/')
    train_gt_imgs_dir = os.path.join(train_dir, 'gt_imgs/')
    val_imgs_dir = os.path.join(val_dir, 'imgs/')
    val_gt_imgs_dir = os.path.join(val_dir, 'gt_imgs/')
    test_imgs_dir = os.path.join(test_dir, 'imgs/')
    test_gt_imgs_dir = os.path.join(test_dir, 'gt_imgs/')
    os.makedirs(train_imgs_dir, exist_ok=True)
    os.makedirs(train_gt_imgs_dir, exist_ok=True)
    os.makedirs(val_imgs_dir, exist_ok=True)
    os.makedirs(val_gt_imgs_dir, exist_ok=True)
    os.makedirs(test_imgs_dir, exist_ok=True)
    os.makedirs(test_gt_imgs_dir, exist_ok=True)

    # get list of image filenames
    img_filenames = os.listdir(imgs_dir)

    # shuffle filenames
    random.shuffle(img_filenames)

    # get number of images and splits
    num_images = len(img_filenames)
    num_train = int(train_size * num_images)
    num_val = int(val_size * num_images)
    num_test = num_images - num_train - num_val

    # split image filenames into train, val, test sets
    train_filenames = img_filenames[:num_train]
    val_filenames = img_filenames[num_train:num_train+num_val]
    test_filenames = img_filenames[num_train+num_val:]

    # copy image files to train, val, test directories and corresponding gt files
    for split, imgs_split_dir, gt_imgs_split_dir, filenames in zip(
            [train_dir, val_dir, test_dir],
            [train_imgs_dir, val_imgs_dir, test_imgs_dir],
            [train_gt_imgs_dir, val_gt_imgs_dir, test_gt_imgs_dir],
            [train_filenames, val_filenames, test_filenames]
        ):
        for filename in filenames:
            # copy image file from imgs_dir to imgs_split_dir
            src_img = os.path.join(imgs_dir, filename)
            dst_img = os.path.join(imgs_split_dir, filename)
            shutil.copyfile(src_img, dst_img)
            # copy gt file with the same filename from gt_imgs_dir to gt_imgs_split_dir
            src_gt = os.path.join(gt_imgs_dir, filename)
            dst_gt = os.path.join(gt_imgs_split_dir, filename)
            shutil.copyfile(src_gt, dst_gt)

    return train_dir, val_dir, test_dir



            


if __name__ == '__main__':
    # 1. remove bad from good
    # keep_dir = "F:/binary_classifier_keep"
    # reject_dir = "F:/binary_classifier_reject"
    # delete_matching_files(keep_dir, reject_dir)

    # 2. combine all subdirs into the training dir
    # src_dir = "F:/binary_classifier_keep"
    # dest_dir = "F:/binary_classifier_training/keep"
    # copy_all_images_to_new_dir(src_dir, dest_dir)

    # src_dir = "E:\TEMP_ruler\Rulers_ByType"
    # dest_dir = "F:/binary_classifier_training_original_img/imgs"
    # copy_all_images_to_new_dir(src_dir, dest_dir)
    # 3. 
    # src_dir = "F:/binary_classifier_training_original_img/gt_imgs"
    # all_imgs = 'F:/binary_classifier_training_original_img/all_imgs'
    # dest_dir = "F:/binary_classifier_training_original_img/imgs"
    # match_DocEnTR_binary_with_parent_rgb(src_dir, all_imgs, dest_dir)

    # 4. 
    img_dir = 'F:/binary_classifier_training_original_img/DIBCOSETS/RULER/imgs'
    gt_dir = 'F:/binary_classifier_training_original_img/DIBCOSETS/RULER/gt_imgs'
    output_dir = 'F:/binary_classifier_training_original_img/DIBCOSETS/RULER/'
    split_train_val_test(img_dir, gt_dir, output_dir, train_size=0.8, val_size=0.1, test_size=0.1)
