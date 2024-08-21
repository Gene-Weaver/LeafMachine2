from PIL import Image, ImageDraw, ImageFont
import math

class ContourVisualizer:
    def __init__(self, filepath, path_out, resolution=(800, 800), dpi=300, do_show=True, do_save=True, plot_pts_size=5, text_size=15,
                 path_original_txt=None, path_out_overlay=None, path_original_image=None):
        self.filepath = filepath
        self.path_out = path_out
        self.resolution = resolution
        self.dpi = dpi
        self.do_show = do_show
        self.do_save = do_save
        self.plot_pts_size = plot_pts_size
        self.text_size = text_size
        self.angle = None
        self.top_most = None
        self.bottom_most = None
        self.predicted_tip = None
        self.predicted_base = None
        self.contour_points = []
        self.bbox = None
        self.path_original_txt = path_original_txt
        self.path_out_overlay = path_out_overlay
        self.path_original_image = path_original_image

    def read_data(self):
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Parse the txt file according to the structure you've provided
        self.original_height = int(lines[0].strip())
        self.original_width = int(lines[1].strip())
        self.original_cf = float(lines[2].strip())
        self.max_extent = float(lines[3].strip())
        self.x_min = float(lines[4].strip())
        self.y_min = float(lines[5].strip())
        self.angle = float(lines[6].strip())

        self.top_most = tuple(map(float, lines[7].strip().split(',')))
        self.bottom_most = tuple(map(float, lines[8].strip().split(',')))
        self.predicted_tip = tuple(map(float, lines[9].strip().split(',')))
        self.predicted_base = tuple(map(float, lines[10].strip().split(',')))

        self.contour_points = [tuple(map(float, line.strip().split(','))) for line in lines[11:]]

    def read_original_data(self):
        with open(self.path_original_txt, 'r') as f:
            lines = f.readlines()

        # Read the original height, width, and conversion factor
        self.original_height = int(lines[0].strip())
        self.original_width = int(lines[1].strip())
        self.original_cf = float(lines[2].strip())

        # Read the bounding box from the original image
        self.bbox = tuple(map(int, lines[3:7]))

        # Read the original angle
        self.angle_original = float(lines[7].strip())

        # Read the original contour points
        self.original_contour_points = [tuple(map(float, line.strip().split(','))) for line in lines[8:]]

    def rotate_point(self, point, angle, origin=(0, 0)):
        """ Rotate a point counterclockwise by a given angle around a given origin. """
        ox, oy = origin
        px, py = point

        angle_rad = math.radians(angle)
        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
        return qx, qy

    def scale_and_translate(self, point, scale, translation):
        return (point[0] * scale[0] + translation[0], point[1] * scale[1] + translation[1])

    def visualize(self):
        width, height = self.resolution
        original_image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(original_image)
        scale = (width / 2.0, height / 2.0)
        translation = (width / 2.0, height / 2.0)

        # Scale the contour points
        scaled_contour = [self.scale_and_translate(p, scale, translation) for p in self.contour_points]
        draw.line(scaled_contour + [scaled_contour[0]], fill='black', width=2)

        font = ImageFont.load_default()

        # Scale and draw significant points
        scaled_top_most = self.scale_and_translate(self.top_most, scale, translation)
        scaled_bottom_most = self.scale_and_translate(self.bottom_most, scale, translation)
        scaled_predicted_tip = self.scale_and_translate(self.predicted_tip, scale, translation)
        scaled_predicted_base = self.scale_and_translate(self.predicted_base, scale, translation)

        draw.ellipse([scaled_top_most[0] - self.plot_pts_size, scaled_top_most[1] - self.plot_pts_size, scaled_top_most[0] + self.plot_pts_size, scaled_top_most[1] + self.plot_pts_size], fill='red')
        draw.ellipse([scaled_bottom_most[0] - self.plot_pts_size, scaled_bottom_most[1] - self.plot_pts_size, scaled_bottom_most[0] + self.plot_pts_size, scaled_bottom_most[1] + self.plot_pts_size], fill='blue')
        draw.ellipse([scaled_predicted_tip[0] - self.plot_pts_size, scaled_predicted_tip[1] - self.plot_pts_size, scaled_predicted_tip[0] + self.plot_pts_size, scaled_predicted_tip[1] + self.plot_pts_size], fill='green')
        draw.ellipse([scaled_predicted_base[0] - self.plot_pts_size, scaled_predicted_base[1] - self.plot_pts_size, scaled_predicted_base[0] + self.plot_pts_size, scaled_predicted_base[1] + self.plot_pts_size], fill='orange')

        draw.text((scaled_top_most[0] + 10, scaled_top_most[1] + 10), 'Top-most', fill='red', font=font)
        draw.text((scaled_bottom_most[0] + 10, scaled_bottom_most[1] + 10), 'Bottom-most', fill='blue', font=font)
        draw.text((scaled_predicted_tip[0] + 10, scaled_predicted_tip[1] - 10), 'Predicted Tip', fill='green', font=font)
        draw.text((scaled_predicted_base[0] + 10, scaled_predicted_base[1] - 10), 'Predicted Base', fill='orange', font=font)

        if self.do_save:
            original_image.save(self.path_out, dpi=(self.dpi, self.dpi))

        if self.do_show:
            original_image.show()

    def overlay_on_original(self):
        self.read_original_data()

        # Load the original image
        original_image = Image.open(self.path_original_image)
        draw_overlay = ImageDraw.Draw(original_image)

        # Calculate the middle point of the bounding box for rotation
        middle_x = self.bbox[0] + (self.bbox[2] - self.bbox[0]) / 2
        middle_y = self.bbox[1] + (self.bbox[3] - self.bbox[1]) / 2

        # Rotate the contour points around the middle of the bounding box by the original angle
        rotated_contour = [self.rotate_point(p, -self.angle_original, origin=(middle_x, middle_y)) for p in self.original_contour_points]

        # Translate the rotated contour points by the top-left corner of the bounding box
        translated_contour = [(p[0] + self.bbox[0], p[1] + self.bbox[1]) for p in rotated_contour]

        # Flip the y-coordinates to account for the top-left origin in PIL
        translated_contour = [(p[0], self.original_height - p[1]) for p in translated_contour]

        # Draw the contour on the original image
        draw_overlay.line(translated_contour + [translated_contour[0]], fill='red', width=2)

        if self.do_save and self.path_out_overlay:
            original_image.save(self.path_out_overlay, dpi=(self.dpi, self.dpi))

        if self.do_show:
            original_image.show()

    @staticmethod
    def main():
        path_txt = 'D:/Dropbox/LeafMachine2/demo/demo_output/test_run_2024_08_16__01-23-57/Keypoints/Simple_Labels/LM_Validate_LowRes_1__L__21-1040-447-1670.txt'
        path_original_txt = 'D:/Dropbox/LeafMachine2/demo/demo_output/test_run_2024_08_16__01-23-57/Keypoints/Simple_Labels_Original/LM_Validate_LowRes_1__L__21-1040-447-1670.txt'

        path_out = 'D:/Dropbox/LeafMachine2/demo/demo_output/test_run_2024_08_16__01-23-57/Keypoints/Simple_Labels/LM_Validate_LowRes_1__L__21-1040-447-1670.png'
        path_out_overlay = 'D:/Dropbox/LeafMachine2/demo/demo_output/test_run_2024_08_16__01-23-57/Keypoints/Simple_Labels_Original/LM_Validate_LowRes_1__L__21-1040-447-1670.png'
        
        path_original_image = 'D:/Dropbox/LeafMachine2/demo/demo_images/LM_Validate_LowRes_1.jpg'
        
        resolution = (1024, 1024)
        dpi = 300
        do_show = True
        do_save = True
        plot_pts_size = 10
        text_size = 35

        visualizer = ContourVisualizer(path_txt, path_out, resolution, dpi, do_show, do_save, plot_pts_size, text_size,
                                       path_original_txt=path_original_txt, path_out_overlay=path_out_overlay, path_original_image=path_original_image)
        visualizer.read_data()
        visualizer.visualize()
        visualizer.overlay_on_original()

if __name__ == "__main__":
    ContourVisualizer.main()
