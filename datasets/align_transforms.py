# Custom "Dataset" class for LAB
# Author: Beier ZHU
# Date: 2019/07/25

import numpy as np
import random
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose
import numbers
import matplotlib.pyplot as plt
from datasets.landmarks_to_heatmap import landmarks_to_boundary_heatmap, landmarks_to_landmark_heatmap
from datasets.procrustes import procrustes


class RandomBlur(object):
    def __call__(self, sample):
        random_blur = random.uniform(0.,1.)
        if random_blur < 0.4:
            image = sample['image']

            if random_blur > 0.2: # Gaussian Blur
                image = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=2, sigmaY=2)
            else:
                angle = random.randint(0, 180)
                image = self.motion_blur(image, ksize=5, angle=angle)

            sample['image'] = image
        return sample

    def motion_blur(self, src, ksize=5, angle=45):
        src = np.array(src)
        R = cv2.getRotationMatrix2D((ksize/2 - 0.5, ksize/2 - 0.5), angle, 1)
        kernel = np.diag(np.ones(ksize)) / ksize
        kernel = cv2.warpAffine(kernel, R, (ksize, ksize))
        dst = cv2.filter2D(src, -1, kernel)
        cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
        dst = np.array(dst, dtype=np.uint8)
        return dst

class RandomOcclude(object):
    def __init__(self, img_size, aspect_ratio=[0.4, 2.5], occ_ratio=0.2):
        self.img_size = img_size
        self.aspect_ratio = aspect_ratio
        self.occ_area = occ_ratio*img_size*img_size

    def _refine_rect(self, lt, rb):
        img_size = self.img_size
        l, t = lt
        r, b = rb
        assert l < r and t < b

        l = 0 if l < 0 else l
        t = 0 if t < 0 else t
        r = img_size if r > img_size else r
        b = img_size if b > img_size else b

        return (l, t), (r, b)

    def __call__(self, sample):
        random_occ = random.uniform(0., 1.)
        if random_occ > 0.5:
            x_c = random.uniform(0., self.img_size)
            y_c = random.uniform(0., self.img_size)
            aspect_ratio = random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
            # w*h = occ_area; w/h = aspect_ratio
            w = (aspect_ratio*self.occ_area)**(0.5)
            h = w / aspect_ratio

            l = int(x_c - w/2)
            r = int(x_c + w/2)
            t = int(y_c - h/2)
            b = int(y_c + h/2)

            (l, t), (r, b) = self._refine_rect((l, t), (r, b))
            image = sample['image']
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (l, t), (r, b), color=color, thickness=-1)
            sample['image'] = image

        return sample

class GuidedAffine(object):
    """Affine transformation of the image according to guided pose, typically mean pose, and pose per se
    Args:
        guided_pose (numpy array of shape (2*#landmarks)x 1 ): guided pose, typically mean pose
    Description:
        Given guided pose and original pose, function `procrustes` will calculate an affine matrix,
        which attemps to project the original pose to guided pose. The affined matrix then appied to
        original image 

    """
    def __init__(self, guided_pose, border_value=(127,127,127)):
        self.guided_pose = guided_pose
        self.border_value = border_value

    @staticmethod
    def _pointsAffine(points, matrix): 
        matrix = matrix.copy()
        matrix[0,1] *= -1
        matrix[1,0] *= -1

        points = points.reshape(-1, 2)
        points = np.matmul(points, matrix[:,:2]) 
        points += np.transpose(matrix[:,2])   
        points = points.reshape(-1)

        return points

    def __call__(self, sample):
        # using predicted landmarks and guided pose to calculate affine matrix
        # using the affine matrix to warp image and landmarks ground true
        image_origin = sample['image_origin']
        landmark_origin = sample['landmarks_origin']
        landmark_pred = sample['landmarks_pred']
        matrix = procrustes(self.guided_pose, landmark_pred, scaling=True, reflection=False)
        sample['landmarks'] = GuidedAffine._pointsAffine(landmark_origin, matrix)
        sample['image'] = cv2.warpAffine(image_origin, matrix, (image_origin.shape[1], image_origin.shape[0]),\
                                         cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue=self.border_value)

        return sample

class ToLandmarkHeatmap(object):
    """
    Transform landmarks to multi-channel landmark heatmaps
    """

    def __init__(self, heatmap_size=64, label_size=256, sigma=1):
        self.heatmap_size = heatmap_size
        self.label_size = label_size
        self.sigma = sigma

    def __call__(self, sample):
        landmarks = sample['landmarks']
        sample['heatmap'] = landmarks_to_landmark_heatmap(points=landmarks,
                                                          heatmap_size=self.heatmap_size,
                                                          label_size=self.label_size,
                                                          sigma=self.sigma)
        return sample
        
class ToBoundaryHeatmap(object):
    """
    Transform landmarks to multi-channel boundary heatmaps
    """
    def __init__(self, heatmap_size=64, label_size=256, sigma=1):
        if isinstance(heatmap_size, int):
            heatmap_size = (heatmap_size, heatmap_size)

        if isinstance(label_size, int):
            label_size = (label_size, label_size)

        self.heatmap_size = heatmap_size
        self.label_size = label_size
        self.sigma = sigma

    def __call__(self, sample):
        landmarks = sample['landmarks']
        sample['boundary'] = landmarks_to_boundary_heatmap(points=landmarks, 
                                                           arrange_mode='global',
                                                           heatmap_size=self.heatmap_size,
                                                           label_size=self.label_size, 
                                                           sigma=self.sigma)
        return sample

def _get_corr_list(num_pts):
    """Show indices for landmarks mirror"""
    assert num_pts in [29, 68, 98, 106], '#landmarks should be 29, 68, 98 or 106, but got {}'.format(num_pts)
    if num_pts == 29:
        corr_list = [0, 1, 4, 6, 2, 3, 5, 7, 8, 9, 13, 15, 12, 14, 10, 11, 16, 17, 18, 19, 22, 23]
    elif num_pts == 68:
        corr_list = [0, 16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 17, 26, 18, 25, 19, 24, 20, 23, 21, 22, 36,
                     45, 37, 44, 38, 43, 39, 42, 41, 46, 40, 47, 31, 35, 32, 34, 48, 54, 49, 53, 50, 52, 60, 64, 61, 63,
                     67, 65, 59, 55, 58, 56]
    elif num_pts == 98:
        corr_list = [0, 32, 1, 31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, 7, 25, 8, 24, 9, 23, 10, 22, 11, 21, 12, 20, 13, 19,
                     14, 18, 15, 17, 33, 46, 34, 45, 35, 44, 36, 43, 37, 42, 38, 50, 39, 49, 40, 48, 41, 47, 55, 59, 56, 58,
                     60, 72, 61, 71, 62, 70, 63, 69, 64, 68, 65, 75, 66, 74, 67, 73, 76, 82, 77, 81, 78, 80, 88, 92, 89, 91,
                     95, 93, 87, 83, 86, 84, 96, 97]
    elif num_pts == 106:
        corr_list = [0, 32, 1, 31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, 7, 25, 8, 24, 9, 23, 10, 22, 11, 21, 12, 20, 13,
                     19,
                     14, 18, 15, 17, 33, 42, 34, 41, 35, 40, 36, 39, 37, 38, 64, 71, 65, 70, 66, 69, 67, 68, 52, 61, 53,
                     60,
                     72, 75, 54, 59, 55, 58, 56, 63, 73, 76, 57, 62, 74, 77, 104, 105, 78, 79, 80, 81, 82, 83, 47, 51,
                     48, 50,
                     84, 90, 96, 100, 85, 89, 86, 88, 95, 91, 94, 92, 97, 99, 103, 101]
    

    corr_list = np.array(corr_list, dtype=np.uint8).reshape(-1, 2)
    return corr_list

def _get_affine_matrix(center, angle, translations, zoom, shear, do_mirror=False):
    """Compute affine matrix from affine transformation"""
    # Rotation & scale
    matrix = cv2.getRotationMatrix2D(center, angle, zoom).astype(np.float64)
    # translate
    matrix[0, 2] += translations[0] * zoom
    matrix[1, 2] += translations[1] * zoom

    mirror_flag = False
    if do_mirror:
        mirror_rng = random.uniform(0., 1.)
        if mirror_rng > 0.5:
            mirror_flag = True
            matrix[0, 0] = -matrix[0, 0]
            matrix[0, 1] = -matrix[0, 1]
            matrix[0, 2] = (center[0] + 0.5) * 2.0 - matrix[0, 2]

    return matrix, mirror_flag


class Normalize(object):
    """
    Normalize the image channel-wise for each input image,
    the mean and std. are calculated from each channel of the given image
    """

    def __call__(self, sample, type='z-score'):        
        image, landmarks = sample['image'], sample['landmarks']

        if len(image.shape) == 2:
            #gray img
            if type == 'z-score':
                mean, std = cv2.meanStdDev(image)
                if std < 1e-6: std = 1.
                image = (image - mean) / std
                image = image[:, :, None].astype(np.float32)

            else:
                # followed by ToTensor, will divide pixel value by 255.
                image = image[:, :, None]

        if len(image.shape) == 3:
            #color img
            if type == 'z-score':
                mean, std = cv2.meanStdDev(image)
                mean, std = mean[:,0], std[:,0]
                std = np.where(std < 1e-6, 1, std)
                image = (image - mean)/std
                image = image.astype(np.float32)

        sample['image'] = image
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        sample['image'] = img
        sample['landmarks'] = landmarks
        return sample


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h, left: left + new_w]

        for i in range(int(len(landmarks) / 2)):
            landmarks[2 * i] -= left
            landmarks[2 * i + 1] -= top

        sample['image'] = image
        sample['landmarks'] = landmarks
        sample['image_cv2'] = image.astype(np.uint8)

        return sample


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False,
                 fillcolor=0, mirror=False, corr_list=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.mirror = mirror
        self.corr_list = corr_list

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            if scale_ranges[0] == scale_ranges[1]:
                scale = scale_ranges[0]
            else:
                scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        image, landmarks = sample['image'], sample['landmarks']
        # print('before affine transform', len(landmarks))
        h, w = image.shape[:2]
        center = (w/2 - 0.5, h/2 - 0.5)
        angle, translations, zoom, shear = \
            self.get_params(self.degrees, self.translate, self.scale, self.shear, [h, w])

        matrix, mirrored = _get_affine_matrix(center, angle, translations, zoom, shear, self.mirror)
        # src = np.array(image).astype(np.uint8)
        # src = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dst = cv2.warpAffine(image, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                             borderValue=(127, 127, 127))
        # landmarks tranformation
        matrix = np.resize(matrix, (6,))
        points = np.resize(np.array(landmarks).copy(), (int(len(landmarks) / 2), 2))

        for i in range(len(landmarks)//2):
            x, y = points[i, :]
            x_new = matrix[0] * x + matrix[1] * y + matrix[2]
            y_new = matrix[3] * x + matrix[4] * y + matrix[5]

            landmarks[2 * i] = x_new
            landmarks[2 * i + 1] = y_new

        if mirrored:
            # reorder index of landmarks after mirroring
            for k in range(self.corr_list.shape[0]):
                temp_x = landmarks[2 * self.corr_list[k, 0]]
                temp_y = landmarks[2 * self.corr_list[k, 0] + 1]
                landmarks[2 * self.corr_list[k, 0]], landmarks[2 * self.corr_list[k, 0] + 1] = \
                    landmarks[2 * self.corr_list[k, 1]], landmarks[2 * self.corr_list[k, 1] + 1]
                landmarks[2 * self.corr_list[k, 1]], landmarks[2 * self.corr_list[k, 1] + 1] = temp_x, temp_y

        sample['image'] = dst
        sample['landmarks'] = landmarks
        return sample


class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        image = sample['image']
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sample['image'] = gray_image
        return sample


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image (scale by 1/255).
        """
        image = sample['image']
        image = TF.to_tensor(image)

        sample['image'] = image
        return sample


