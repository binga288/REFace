# copy from https://github.com/rotemtzaban/STIT/blob/main/utils/alignment.py

import PIL
import PIL.Image
# import dlib
# import face_alignment
import numpy as np
import scipy
import scipy.ndimage
import skimage.io as io
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import cv2
import dlib


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def get_landmark(filepath, predictor, detector=None, fa=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    import dlib
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    if isinstance(filepath, str):
        img = dlib.load_rgb_image(filepath)
    elif isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
    elif isinstance(filepath, np.ndarray):
        img = filepath
    else:
        raise ValueError("Unsupported input type: {}".format(type(filepath)))

    if detector is None:
        detector = dlib.get_frontal_face_detector()
    dets = detector(img)

    for k, d in enumerate(dets):
        
        # # 上下扩张50个pixel, 左右扩张20个pixel
        # d = dlib.rectangle(left=max(0, d.left()-20), top=max(0, d.top()-50), right=min(d.right()+20, img.shape[1]), bottom=min(d.bottom(), img.shape[0]))
    
        # (x, y, w, h) = rect_to_bb(d)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        shape = predictor(img, d)
        
        # shape_np = shape_to_np(shape)
        # for (x, y) in shape_np:
        #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        break
    else:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath_or_image, predictor, output_size, detector=None,
               enable_padding=False, scale=1.0):
    """
    :param filepath: str
    :return: PIL Image
    """

    c, x, y = compute_transform(filepath_or_image, predictor, detector=detector,
                                scale=scale)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    img = crop_image(filepath_or_image, output_size, quad, enable_padding=enable_padding)

    # Return aligned image.
    return img


def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    else:
        img = PIL.Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)  # (left, upper, right, lower)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img


def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        raise Exception(f'Did not detect any faces in image: {filepath}')
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y


def crop_faces(IMAGE_SIZE, files, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False):
    if use_fa:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
    else:
        # import dlib
        fa = None
        predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()

    cs, xs, ys = [], [], []
    for  path in tqdm(files):
        c, x, y = compute_transform(path, predictor, detector=detector,
                                    scale=scale, fa=fa)
        cs.append(c)
        xs.append(x)
        ys.append(y)

    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    crops, orig_images = crop_faces_by_quads(IMAGE_SIZE, files, quads)

    return crops, orig_images, quads

def crop_faces_from_image(IMAGE_SIZE, frame, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False):
    if use_fa:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = None
        detector = None
    else:
        import dlib
        fa = None
        predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()

    cs, xs, ys = [], [], []
    path = frame
    c, x, y = compute_transform(path, predictor, detector=detector,
                                scale=scale, fa=fa)
    cs.append(c)
    xs.append(x)
    ys.append(y)

    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    crops, orig_images = crop_faces_by_quads(IMAGE_SIZE, [path], quads)

    return crops, orig_images, quads

def crop_faces_by_quads(IMAGE_SIZE, files, quads):
    orig_images = []
    crops = []
    for quad, path in tqdm(zip(quads, files), total=len(quads)):
        if isinstance(path, str):
            orig_image = Image.open(path)
        elif isinstance(path, np.ndarray):
            orig_image = Image.fromarray(path)
        elif isinstance(path, PIL.Image.Image):
            orig_image = path
        else:
            raise ValueError("Unsupported path type")
        crop = crop_image(orig_image, IMAGE_SIZE, quad.copy())  # Pass orig_image to crop_image
        orig_images.append(orig_image)
        crops.append(crop)
    return crops, orig_images


def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)
