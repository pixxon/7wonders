from albumentations import Compose, ShiftScaleRotate, RandomBrightnessContrast, KeypointParams
import numpy, cv2

def augment(image, polylines):
    augmentation = Compose(
        [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.8, 0.5), rotate_limit=180, p=1, border_mode=cv2.BORDER_CONSTANT),
#            RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.4),
        ],
        keypoint_params=KeypointParams(format='xy', remove_invisible=False)
    )
    
    augmented = augmentation(image=image, keypoints=polylines[:,1:].reshape(-1, 2))
    new_polylines = polylines.copy()
    new_polylines[:,1:] = augmented['keypoints'].reshape(-1, 8)
    return augmented['image'], new_polylines

def stack(image1, polylines1, image2, polylines2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    shift = int(max(numpy.concatenate(polylines1[:,2::2])))
    image = numpy.zeros((max(h1, h2 + shift), max(w1, w2), 3), numpy.uint8)

    image[:h1, :w1, :3] = image1
    image[shift:shift + h2, :w2, :3] = image2

    new_polylines2 = polylines2.copy()
    new_polylines2[:,2::2] += shift

    return image, numpy.concatenate((polylines1, new_polylines2))

def embed(background, image, polylines):
    background = background.copy()

    bgh, bgw, _ = background.shape
    ih, iw, _ = image.shape

    xoff = round((bgw - iw) / 2)
    yoff = round((bgh - ih) / 2)

    background[yoff:yoff + ih, xoff:xoff + iw] = image

    polylines = polylines.copy()
    polylines[:,1::2] += xoff
    polylines[:,2::2] += yoff

    return background, polylines

def crop(image, polylines):
    polylines = polylines.copy()
    polylines[:,:, 1] -= 500
    polylines[:,:, 0] -= 500
    return image[500:1500, 500:1500],  polylines

def resize(image, old_keypoints, height, width):
    old_height, old_width = image.shape[:2]
    image = cv2.resize(image, (width, height))
    keypoints = old_keypoints.copy()
    keypoints[:,1::2] *= (width / old_width)
    keypoints[:,2::2] *= (height / old_height)
    return image, keypoints
