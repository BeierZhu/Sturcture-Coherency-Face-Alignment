#!/usr/bin/python3
import numpy as np
from datasets.vec2 import Vec2, distance
import cv2
from scipy.interpolate import CubicSpline
import itertools

boundary_individual_index_68pts = list(range(13))

boundary_global_index_68pts = [0]*13

boundary_group_index_68pts = [0, 1, 2, 3, 3, 1, 1, 2, 2, 4, 4, 4, 4]
'''                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
                              contour      0                      0
                              left eye     1, 5, 6                1
                              right eye    2, 7, 8                2
                              nose         3, 4                   3   
                              mouth        9, 10, 11, 12          4
'''

boundary_index_68pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],   # contour               0
    [17, 18, 19, 20, 21],                                         # left top eyebrow      1
    [22, 23, 24, 25, 26],                                         # right top eyebrow     2
    [27, 28, 29, 30],                                             # nose bridge           3
    [31, 32, 33, 34, 35],                                         # nose tip              4 
    [36, 37, 38, 39],                                             # left top eye          5
    [39, 40, 41, 36],                                             # left bottom eye       6
    [42, 43, 44, 45],                                             # right top eye         7
    [45, 46, 47, 42],                                             # right bottom eye      8
    [48, 49, 50, 51, 52, 53, 54],                                 # up up lip             9
    [60, 61, 62, 63, 64],                                         # up bottom lip        10 
    [60, 67, 66, 65, 64],                                         # bottom up lip        11
    [48, 59, 58, 57, 56, 55, 54]                                  # bottom bottom lip    12
]

boundary_individual_index_98pts = list(range(15))

boundary_global_index_98pts = [0]*15

boundary_group_index_98pts = [0, 1, 2, 3, 3, 1, 1, 2, 2, 4,  4,  4,  4,  1,  2]
'''                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
                             contour                0                         0
                             left eye               1, 5, 6, 13               1
                             right eye              2, 7, 8, 14               2
                             nose                   3, 4                      3
                             mouth                  9, 10, 11, 12             4
'''


boundary_index_98pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # contour               0
    [33, 34, 35, 36, 37],                                         # left top eyebrow      1
    [42, 43, 44, 45, 46],                                         # right top eyebrow     2
    [51, 52, 53, 54],                                             # nose bridge           3
    [55, 56, 57, 58, 59],                                         # nose tip              4
    [60, 61, 62, 63, 64],                                         # left top eye          5
    [60, 67, 66, 65, 64],                                         # left bottom eye       6
    [68, 69, 70, 71, 72],                                         # right top eye         7
    [68, 75, 74, 73, 72],                                         # right bottom eye      8
    [76, 77, 78, 79, 80, 81, 82],                                 # up up lip             9
    [88, 89, 90, 91, 92],                                         # up bottom lip        10 
    [88, 95, 94, 93, 92],                                         # bottom up lip        11 
    [76, 87, 86, 85, 84, 83, 82],                                 # bottom bottom lip    12 
    [33, 41, 40, 39, 38],                                         # left bottom eyebrow  13 
    [50, 49, 48, 47, 46]                                          # right bottom eyebrow 14 
]

boundary_individual_index_106pts = list(range(15))

boundary_global_index_106pts = [0]*15

boundary_group_index_106pts = [0, 1, 2, 3, 3, 1, 1, 2, 2, 4,  4,  4,  4,  1,  2]
'''                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
                               contour                0                         0
                               left eye               1, 5, 6, 13               1
                               right eye              2, 7, 8, 14               2
                               nose                   3, 4                      3
                               mouth                  9, 10, 11, 12             4
'''

boundary_index_106pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # contour               0
    [33, 34, 35, 36, 37],                                         # left top eyebrow      1
    [38, 39, 40, 41, 42],                                         # right top eyebrow     2
    [43, 44, 45, 46],                                             # nose bridge           3
    [80, 82, 47, 48, 49, 50, 51, 83, 81],                         # nose tip              4
    [52, 53, 72, 54, 55],                                         # left top eye          5
    [52, 57, 73, 56, 55],                                         # left bottom eye       6
    [58, 59, 75, 60, 61],                                         # right top eye         7
    [58, 63, 76, 62, 61],                                         # right bottom eye      8
    [84, 85, 86, 87, 88, 89, 90],                                 # up up lip             9
    [96, 97, 98, 99, 100],                                        # up bottom lip        10 
    [96, 103, 102, 101, 100],                                     # bottom up lip        11 
    [84, 95, 94, 93, 92, 91, 90],                                 # bottom bottom lip    12 
    [33, 64, 65, 66, 67],                                         # left bottom eyebrow  13 
    [68, 69, 70, 71, 42]                                          # right bottom eyebrow 14 
]

def _floats2Vecs(np_points):
    return [Vec2(x, y) for x, y in zip(np_points[::2], np_points[1::2])]
    
def _fit_points_in_boundary(points):
    num_pts = len(points)    
    assert num_pts in [68, 98, 106], "#landmarks should be 68, 98 or 106, but got %d" %num_pts

    if num_pts == 68: boundary_index = boundary_index_68pts
    if num_pts == 98: boundary_index = boundary_index_98pts
    if num_pts == 106: boundary_index = boundary_index_106pts

    points_list = [[]]*len(boundary_index)
    for i, value in enumerate(boundary_index):
        points_list[i] = [points[j] for j in value]

    return points_list

def _fit_boundary_in_part(mode, num_pts):
    assert num_pts in [68, 98, 106]
    if mode == 'individual': 
        if num_pts == 68:
            part_index = boundary_individual_index_68pts
        elif num_pts == 98:
            part_index = boundary_individual_index_98pts
        else:
            part_index = boundary_individual_index_106pts
    elif mode == 'global': 
        if num_pts == 68:
            part_index = boundary_global_index_68pts
        elif num_pts == 98:
            part_index = boundary_global_index_98pts
        else:
            part_index = boundary_global_index_106pts
    elif mode == 'group': 
        if num_pts == 68:
            part_index = boundary_group_index_68pts
        elif num_pts == 98:
            part_index = boundary_group_index_98pts
        else:
            part_index = boundary_group_index_106pts
    else: raise Exception('Invalid mode, should be individual, global or group', mode)

    return part_index

def _filter_valid_point(points, epsilon):
    num_pts = len(points)
    assert num_pts > 2, "#points should greater than 2, but got %d." %num_pts

    # TODO: need better algorithm
    valid_pts = [points[i] for i in range(1, num_pts) if distance(points[i], points[i-1]) > epsilon]
    valid_pts.insert(0, points[0])
    # assert len(valid_pts) > 2, "#valid points should greater than 2, but got %d" %len(valid_pts)
    if len(valid_pts) <= 2:
        print("[WARNING]#valid points should greater than 2, but got %d" %len(valid_pts)) 

    return valid_pts


def cubic_spline(points, sample_rate=10, epsilon=0.001):
    points = _filter_valid_point(points, epsilon)
    if len(points) <= 2:
        return [] 
    num_pts = len(points)
    num_seg = num_pts - 1

    s_list = [distance(points[i], points[i+1]) for i in range(num_seg)]
    s_list = [0.] + s_list
    s_list = list(itertools.accumulate(s_list))

    x_list = [pt.x for pt in points]
    y_list = [pt.y for pt in points]

    cs_x = CubicSpline(s_list, x_list)
    cs_y = CubicSpline(s_list, y_list)

    s_interpolated = np.arange(0, s_list[-1], (s_list[-1]/(num_pts*sample_rate)))

    x_interpolated = cs_x(s_interpolated)
    y_interpolated = cs_y(s_interpolated)

    curve = []
    for x,y in zip(x_interpolated, y_interpolated):
        curve.append(Vec2(x,y))

    return curve

def landmarks_to_boundary_heatmap(points, arrange_mode, heatmap_size=(384, 384), label_size=(256, 256), sigma=1):
    points = _floats2Vecs(points)
    # reshape points label
    points = list(map(lambda pt: pt*Vec2(heatmap_size[1]/label_size[1], heatmap_size[0]/label_size[0]), points))
    num_pts = len(points)

    points_list = _fit_points_in_boundary(points)
    part_list = _fit_boundary_in_part(arrange_mode, num_pts)
    num_boundary = len(points_list)
    num_channels = len(set(part_list))
    # binary boundary map B described in subsec 3.1
    B = np.full((num_channels, heatmap_size[0], heatmap_size[1]), 255, dtype=np.uint8)
    heatmap = np.zeros((num_channels, heatmap_size[0], heatmap_size[1]))

    for i in range(num_boundary):
        points_interpolated = cubic_spline(points_list[i]) # get dense boundary line

        for pt in points_interpolated:
            int_x, int_y = int(round(pt.x)), int(round(pt.y))
            if int_x > 0 and int_x < heatmap_size[1] and int_y > 0 and int_y < heatmap_size[0]:
                B[part_list[i], int_y, int_x] = 0

    for i in range(num_channels):
        D = cv2.distanceTransform(B[i,:,:], cv2.cv2.DIST_L2, cv2.cv2.DIST_MASK_PRECISE)
        D = D.astype(np.float64)

        D_gaussian = np.exp(-1.0 * D**2 / (2.0*sigma**2))
        D_gaussian = np.where(D < 3.0*sigma, D_gaussian, 0)
        maxD, minD = D_gaussian.max(), D_gaussian.min()

        if maxD == minD: 
            D_gaussian = 0
        else:
            D_gaussian = (D_gaussian - minD) / (maxD - minD)

        heatmap[i, :, :] = D_gaussian

    return heatmap

def put_gaussian_map(heatmap_size, point, sigma=1.0):
    """
    Paint gaussian heatmap
    Args:
        heatmap_size (int): size of output heatmap
        point (Vec2): a landmark
        sigma (float): radius
    """
    point.x = int(round(point.x)) # only for inter coordinates input
    point.y = int(round(point.y))
    size = 2 * 3 * sigma + 1

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    radius = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    ret = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    if point.x < 0 or point.x >= heatmap_size or point.y < 0 or point.y >= heatmap_size:
       return ret

    left = max(0, point.x - radius)
    t = max(0, point.y - radius)
    r = min(heatmap_size - 1, point.x + radius)
    b = min(heatmap_size - 1, point.y + radius)

    ml = x0 - min(point.x, radius)
    mt = y0 - min(point.y, radius)
    mr = x0 + min(heatmap_size - 1 - point.x, radius)
    mb = y0 + min(heatmap_size - 1 - point.y, radius)
    l, t, r, b = list(map(int, [left, t, r, b]))
    ml, mt, mr, mb = list(map(int, [ml, mt, mr, mb]))
    ret[t:b + 1, l:r + 1] = g[mt:mb + 1, ml:mr + 1]
    return ret

def landmarks_to_landmark_heatmap(points, heatmap_size=64, label_size=256, sigma=1.):
    points = _floats2Vecs(points)
    # reshape points label
    points = list(map(lambda pt: pt*Vec2(heatmap_size/label_size, heatmap_size/label_size), points))
    num_pts = len(points)

    label = np.zeros((num_pts, heatmap_size, heatmap_size), dtype=np.float32)

    for i, pt in enumerate(points):
        label[i] = put_gaussian_map(heatmap_size, pt, sigma)

    return label

