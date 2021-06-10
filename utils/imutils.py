import numpy as np
import io
import cv2
from PIL import Image

cv2.ocl.setUseOpenCL(False)


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    open_cv_image = np.array(img)
    return open_cv_image


def put_gaussian_map(label_h, label_w, kpt, sigma=1.0):
    """
    paint gauss heatmap

    :param label_h: gauss heatmap height
    :param label_w: gauss heatmap width
    :param kpt: input keypoints position
    :param sigma: radius
    :return:
    """
    size = 2 * 3 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    radius = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    ret = np.zeros((label_h, label_w), dtype=np.float32)
    if kpt[0] < 0 or kpt[0] >= label_w or kpt[1] < 0 or kpt[1] >= label_h:
        return ret

    left = max(0, kpt[0] - radius)
    t = max(0, kpt[1] - radius)
    r = min(label_w - 1, kpt[0] + radius)
    b = min(label_h - 1, kpt[1] + radius)

    ml = x0 - min(kpt[0], radius)
    mt = y0 - min(kpt[1], radius)
    mr = x0 + min(label_w - 1 - kpt[0], radius)
    mb = y0 + min(label_h - 1 - kpt[1], radius)
    l, t, r, b = list(map(int, [left, t, r, b]))
    ml, mt, mr, mb = list(map(int, [ml, mt, mr, mb]))
    ret[t:b + 1, l:r + 1] = g[mt:mb + 1, ml:mr + 1]
    return ret


def crop(image, ctr, box_w, box_h, rot, in_w, in_h,
         padding_val=np.array([123.675, 116.28, 103.53])):
    """
    crop image using input params
    :param image: input_image
    :param ctr: center
    :param box_w: box width
    :param box_h: box height
    :param rot: rotate angle
    :param in_w: the network input width
    :param in_h: the network input height
    :param padding_val: padding value when make border or warpAffine use
    :return:
    """
    pad_w, pad_h = int(box_w / 2), int(box_h / 2)

    # pad image
    pad_mat = cv2.copyMakeBorder(image,
                                 top=pad_h,
                                 bottom=pad_h,
                                 left=pad_w,
                                 right=pad_w,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=padding_val)

    left = ctr[0] - box_w / 2 + pad_w
    top = ctr[1] - box_h / 2 + pad_h
    right = ctr[0] + box_w / 2 + pad_w
    bottom = ctr[1] + box_h / 2 + pad_h
    l, t, r, b = map(int, [left, top, right, bottom])
    image_roi = pad_mat[t:b, l:r, :]
    image_roi = cv2.resize(image_roi, (in_w, in_h))

    # rotate
    rows, cols, channels = image_roi.shape
    assert channels == 3
    pad_ctr = (int(cols / 2), int(rows / 2))
    rot_matrix = cv2.getRotationMatrix2D(pad_ctr, rot, 1)
    ret = cv2.warpAffine(
        image_roi,
        rot_matrix,
        (in_w,
         in_h),
        flags=cv2.INTER_LINEAR,
        borderValue=padding_val)
    return ret


def get_transform(center, w, h, rot, res_w, res_h):
    """
    General image processing functions
    """
    # Generate transformation matrix
    scale_w = float(res_w) / w
    scale_h = float(res_h) / h
    t = np.zeros((3, 3))
    t[0, 0] = scale_w
    t[1, 1] = scale_h
    t[2, 2] = 1
    t[0, 2] = -float(center[0]) * scale_w + .5 * res_w
    t[1, 2] = -float(center[1]) * scale_h + .5 * res_h
    if not rot == 0:
        rot = -rot                  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res_w / 2
        t_mat[1, 2] = -res_h / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def kpt_transform(pts, center, box_w, box_h, rot, res_w, res_h, invert):
    t = get_transform(center, box_w, box_h, rot, res_w, res_h)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pts[0], pts[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def _find_bbox(src):
    h, w, _ = src.shape

    x_c = (w - 1) / 2. 
    y_c = (h - 1) / 2.

    x = np.arange(0, w, 1, float)
    y = x[:, np.newaxis]
    xx = x - x_c
    yy = y - y_c

    d = xx**2 + yy**2

    delta = 10

    mask_b = (src[:,:,0] < 127 + delta) * (src[:,:,0] > 127 - delta)
    mask_g = (src[:,:,1] < 127 + delta) * (src[:,:,1] > 127 - delta)
    mask_r = (src[:,:,2] < 127 + delta) * (src[:,:,2] > 127 - delta)

    mask = ~ (mask_b * mask_g * mask_r)
    mask_lt = (xx < 0) * (yy < 0) * mask
    mask_rb = (xx > 0) * (yy > 0) * mask

    tl = np.argmax(d * mask_lt)
    br = np.argmax(d * mask_rb)

    t, l = divmod(tl, w)
    b, r = divmod(br, w)

    return (t, l), (b, r)

def refine_300W_landmarks(images, landmarks_pred):
    landmarks_refine = np.zeros(shape=landmarks_pred.shape, dtype=landmarks_pred.dtype)

    for i, (image, landmark_pred) in enumerate(zip(images, landmarks_pred)):
        (t, l), (b, r) = _find_bbox(image) 
        for j in range(len(landmark_pred)//2):
            landmarks_refine[i,2*j] = landmark_pred[2*j] if landmark_pred[2*j] > l else l
            landmarks_refine[i,2*j] = landmark_pred[2*j] if landmark_pred[2*j] < r else r
            landmarks_refine[i,2*j+1] = landmark_pred[2*j+1] if landmark_pred[2*j+1] > t else t
            landmarks_refine[i,2*j+1] = landmark_pred[2*j+1] if landmark_pred[2*j+1] < b else b

    return landmarks_refine

