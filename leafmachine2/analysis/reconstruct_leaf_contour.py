import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def main():
    # Read the data from the file
    file_path = 'D:/D_Desktop/reconstruct/AASU_20809714_Asteraceae_Packera_aurea_H__L__261-2901-647-3375.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Load the image
    image_path = 'D:/D_Desktop/reconstruct/AASU_20809714_Asteraceae_Packera_aurea_H__L__261-2901-647-3375.jpg'
    img = Image.open(image_path)
    img_array = np.array(img)

    # Parse the data from the file
    full_size_height = float(lines[0].strip())
    full_size_width = float(lines[1].strip())
    CF = float(lines[2].strip())  # Conversion factor (pixels per mm)
    max_extent = float(lines[3].strip())
    x_min = float(lines[4].strip())
    y_min = float(lines[5].strip())
    angle = float(lines[6].strip())

    # Starting from line 7, parse the points
    points = []
    for line in lines[11:]:
        x_str, y_str = line.strip().split(',')
        x = float(x_str)
        y = float(y_str)
        points.append((x, y))

    # Reverse the transformations
    # Adjust the angle for the image coordinate system
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    original_contour = []
    for x, y in points:
        # Reverse rotation
        x_rot = x * cos_theta + y * sin_theta
        y_rot = -x * sin_theta + y * cos_theta

        # Reverse scaling
        x_scaled = x_rot * max_extent
        y_scaled = y_rot * max_extent

        # Reverse translation
        x_orig = x_scaled + x_min
        y_orig = y_scaled + y_min

        original_contour.append([x_orig, y_orig])

    # Adjust the contour points for image coordinates (PIL starts from the top-left corner)
    original_contour_adjusted = []
    for x_orig, y_orig in original_contour:
        # Flip the y-coordinate to align with the image coordinate system
        y_orig_flipped = -y_orig
        original_contour_adjusted.append([x_orig, y_orig_flipped])

    # Convert to a NumPy array
    original_contour_np = np.array(original_contour_adjusted, dtype=np.float32)

    # For OpenCV functions, the contour needs to be of shape (N, 1, 2)
    contour_cv = original_contour_np.reshape((-1, 1, 2)).astype(np.int32)

    # Calculate area and perimeter in pixels
    area_pixels = cv2.contourArea(contour_cv)
    perimeter_pixels = cv2.arcLength(contour_cv, True)

    # Convert to cm^2 and cm directly (since CF is pixels per cm)
    area_cm2 = area_pixels / (CF ** 2)  # Area in cm²
    perimeter_cm = perimeter_pixels / CF  # Perimeter in cm

    # Print the calculated areas and perimeters
    print(f"Area in pixels: {area_pixels}")
    print(f"Perimeter in pixels: {perimeter_pixels}")
    print(f"Area in cm²: {area_cm2}")
    print(f"Perimeter in cm: {perimeter_cm}")

    # Plot the reconstructed contour over the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_array)
    plt.plot(original_contour_np[:, 0], original_contour_np[:, 1], 'r-', linewidth=2)
    plt.title('Reconstructed Contour Overlayed on Image')
    plt.axis('off')  # Hide the axis
    plt.show()

if __name__ == "__main__":
    main()
