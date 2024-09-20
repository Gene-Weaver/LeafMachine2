def unpack_class_from_components(ProjectSQL, filename, cls, dict_name_yolo, dict_name_location):
    conn = ProjectSQL.conn
    cur = conn.cursor()

    # Get the width and height from the images table
    cur.execute("SELECT width, height FROM images WHERE name = ?", (filename,))
    width, height = cur.fetchone()

    # Retrieve plant annotations for the given filename and class
    cur.execute("SELECT annotation FROM annotations_plant WHERE file_name = ? AND component = 'Detections_Plant_Components'", (filename,))
    plant_annotations = cur.fetchall()

    for annotation in plant_annotations:
        # Process the annotation data to extract bounding box coordinates
        class_index, x_center, y_center, bbox_width, bbox_height = map(float, annotation[0].split(','))

        if int(class_index) == cls:
            x_min = int(x_center * width - (bbox_width * width / 2))
            y_min = int(y_center * height - (bbox_height * height / 2))
            x_max = int(x_center * width + (bbox_width * width / 2))
            y_max = int(y_center * height + (bbox_height * height / 2))

            # Insert the processed bounding box into the correct table (Whole_Leaf_BBoxes or Partial_Leaf_BBoxes)
            cur.execute(f"INSERT INTO {dict_name_location} (file_name, class, x_min, y_min, x_max, y_max) VALUES (?, ?, ?, ?, ?, ?)",
                        (filename, cls, x_min, y_min, x_max, y_max))

    conn.commit()

def crop_images_to_bbox(ProjectSQL, filename, cls, dict_name_cropped, dict_from):
    conn = ProjectSQL.conn
    cur = conn.cursor()

    # Retrieve bounding boxes from the SQL database
    cur.execute(f"SELECT x_min, y_min, x_max, y_max FROM {dict_from} WHERE file_name = ? AND class = ?", (filename, cls))
    bboxes = cur.fetchall()

    # Try to load the image
    try:
        img_path = glob.glob(os.path.join(ProjectSQL.dir_images, f"{filename}.*"))[0]
        img = cv2.imread(img_path)
    except:
        img = None

    if img is None:
        return

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        img_crop = img[y_min:y_max, x_min:x_max]
        loc = '-'.join(map(str, [x_min, y_min, x_max, y_max]))
        crop_name = f"{filename}__{'L' if cls == 0 else 'PL'}__{loc}"

        # Store the cropped image in the SQL database (as a BLOB)
        _, img_encoded = cv2.imencode('.jpg', img_crop)
        cur.execute(f"INSERT INTO {dict_name_cropped} (file_name, crop_name, cropped_image) VALUES (?, ?, ?)", 
                    (filename, crop_name, img_encoded.tobytes()))
    conn.commit()