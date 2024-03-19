import os
import imageio

# Call the function with the path to your directory
# dir_out = convert_cr2_to_jpg('/path/to/your/directory')
def convert_cr2_to_jpg(dir_in, Dirs):
    import rawpy
    # Define the output directory
    dir_out = os.path.join(Dirs.dir_project, 'Cropped_Images', 'temp_jpgs')
    # dir_out_tiff = os.path.join(Dirs.dir_project, 'Cropped_Images', 'cropped_tiffs')
    # Create the output directory if it does not exist
    os.makedirs(dir_out, exist_ok=True)
    # os.makedirs(dir_out_tiff, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(dir_in):
        # Check if this file is a CR2 file
        if filename.lower().endswith('.cr2'):
            # Define the full file path
            filepath = os.path.join(dir_in, filename)
            # Use rawpy to read the CR2 file
            with rawpy.imread(filepath) as raw:
                # Get RGB image
                rgb = raw.postprocess(use_camera_wb=True)
            # Define the output file path
            filepath_out = os.path.join(dir_out, filename[:-4] + '.jpg')
            # Save the image
            imageio.imsave(filepath_out, rgb)
    return dir_out#, dir_out_tiff


# Call the function with the path to your directory
# print(check_image_extensions('/path/to/your/directory'))
def check_image_extensions(dir_in):
    # Define the allowed extensions
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    # Make sure the allowed extensions are all lower case
    allowed_extensions = [ext.lower() for ext in allowed_extensions]

    # Loop through all files in the directory
    for filename in os.listdir(dir_in):
        # Check if this file is an image by looking at the extension
        extension = os.path.splitext(filename)[1].lower()
        if extension and extension not in allowed_extensions:
            return True

    # If no disallowed extensions were found, return False
    return False


def check_image_compliance(cfg, Dirs):

    dir_in = cfg['leafmachine']['project']['dir_images_local']

    has_invalid_images = check_image_extensions(dir_in)

    if has_invalid_images:
        # dir_out, dir_out_tiff = convert_cr2_to_jpg(dir_in)
        dir_out = convert_cr2_to_jpg(dir_in, Dirs)
        cfg['leafmachine']['project']['dir_images_local'] = dir_out
        return cfg, dir_in#, dir_out_tiff
    else:
        return cfg, None