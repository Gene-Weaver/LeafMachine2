'''
These were moved from segment_leaves.py to avoid circular import
'''
import cv2
import numpy as np

def get_largest_polygon(polygons):
    try:
        # polygons = max(polygons, key=len)
        # Keep the polygon that has the most points
        polygons = [polygon for polygon in polygons if len(polygon) == max([len(p) for p in polygons])]
        # convert the list of polygons to a list of contours
        contours = [np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for polygon in polygons]
        # filter the contours to only closed contours
        closed_contours = [c for c in contours if cv2.isContourConvex(c)]
        if len(closed_contours) > 0:
            # sort the closed contours by area
            closed_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)
            # take the largest closed contour
            largest_closed_contour = closed_contours[0]
        else:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_closed_contour = contours[0]
        largest_polygon = [tuple(i[0]) for i in largest_closed_contour]
    except:
        largest_polygon = None
    return largest_polygon

def get_string_indices(strings):
    leaf_strings = [s for s in strings if s.startswith('leaf')]
    petiole_strings = [s for s in strings if s.startswith('petiole')]
    hole_strings = [s for s in strings if s.startswith('hole')]

    if len(leaf_strings) > 0:
        leaf_value = max([int(s.split(' ')[1].replace('%','')) for s in leaf_strings])
        leaf_index = strings.index([s for s in leaf_strings if int(s.split(' ')[1].replace('%','')) == leaf_value][0])
    else:
        leaf_index = None

    if len(petiole_strings) > 0:
        petiole_value = max([int(s.split(' ')[1].replace('%','')) for s in petiole_strings])
        petiole_index = strings.index([s for s in petiole_strings if int(s.split(' ')[1].replace('%','')) == petiole_value][0])
    else:
        petiole_index = None

    if len(hole_strings) > 0:
        hole_indices = [i for i, s in enumerate(strings) if s.startswith('hole')]
    else:
        hole_indices = None

    return leaf_index, petiole_index, hole_indices

def keep_rows(list1, list2, list3, list4, string_indices):
    leaf_index, petiole_index, hole_indices = string_indices
    # All
    if (leaf_index is not None) and (petiole_index is not None) and (hole_indices is not None):
        indices_to_keep = [leaf_index, petiole_index] + hole_indices
    # No holes
    elif (leaf_index is not None) and (petiole_index is not None) and (hole_indices is None):
        indices_to_keep = [leaf_index, petiole_index]
    # Only leaves
    elif (leaf_index is not None) and (petiole_index is None) and (hole_indices is None):
        indices_to_keep = [leaf_index]
    # Only petiole
    elif (leaf_index is None) and (petiole_index is not None) and (hole_indices is None):
        indices_to_keep = [petiole_index]
    # Only hole
    elif (leaf_index is None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep = hole_indices
    # Only petiole and hole
    elif (leaf_index is None) and (petiole_index is not None) and (hole_indices is not None):
        indices_to_keep =  [petiole_index] + hole_indices
    # Only holes and no leaves or petiole
    elif (leaf_index is None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep = hole_indices
    # Only leaves and hole
    elif (leaf_index is not None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep =  [leaf_index] + hole_indices
    else:
        indices_to_keep = None

    # get empty list1 values []
    indices_empty = [i for i, lst in enumerate(list1) if not lst]
    indices_to_keep = [i for i in indices_to_keep if i not in indices_empty]

    if indices_to_keep is not None:
        list1 = [list1[i] for i in indices_to_keep]
        list2 = [list2[i] for i in indices_to_keep]
        list3 = [list3[i] for i in indices_to_keep]
        list4 = [list4[i] for i in indices_to_keep]
        return list1, list2, list3, list4
    else:
        return None, None, None, None