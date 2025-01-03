import albumentations as A
import cv2
import numpy as np
import random

def denormalize_keypoints(keypoints: np.ndarray, image_shape: tuple[int, int]):
    height, width = image_shape[:2]
    denormalized = keypoints.copy()
    denormalized[1::2] *= width
    denormalized[2::2] *= height
    return denormalized

def normalize_keypoints(keypoints: np.ndarray, image_shape: tuple[int, int]):
    height, width = image_shape[:2]
    normalized = keypoints.copy()
    normalized[1::2] /= width
    normalized[2::2] /= height
    return normalized

def read_data(basepath, filename):
    image = cv2.imread(f'{basepath}/images/train/{filename}.jpg')
    resize_ratio = (400 / float(image.shape[0]))
    image = cv2.resize(image, None, fx = resize_ratio, fy = resize_ratio)
    return image, np.array([denormalize_keypoints(polyline, image.shape) for polyline in np.loadtxt(f'{basepath}/labels/train/{filename}.txt').reshape(-1, 9)])

def write_data(image, polylines, basepath, type, filename):
    # image = cv2.polylines(image, [np.array(polyline[1:].reshape(-1, 2), dtype=np.int32) for polyline in polylines], isClosed=True, color=(255, 0, 255), thickness=7)
    cv2.imwrite(f'{basepath}/images/{type}/{filename}.jpg', image)
    np.savetxt(f'{basepath}/labels/{type}/{filename}.txt', [normalize_keypoints(polyline, image.shape) for polyline in polylines.reshape(-1, 9)], '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f')

def augment(image, polylines, count):
    augmentation = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.8, 0.5), rotate_limit=180, p=1, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.4),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    
    for i in range(count):
        augmented = augmentation(image=image, keypoints=polylines[:,1:].reshape(-1, 2))
        tmp = polylines.copy()
        tmp[:,1:] = augmented['keypoints'].reshape(-1, 8)
        yield i, augmented['image'], tmp

def stack(image1, polylines1, image2, polylines2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    shift = int(max(np.concatenate(polylines1[:,2::2])))
    image = np.zeros((max(h1, h2 + shift), max(w1, w2), 3), np.uint8)

    image[:h1, :w1, :3] = image1
    image[shift:shift + h2, :w2, :3] = image2

    new_polylines2 = polylines2.copy()
    new_polylines2[:,2::2] += shift

    return image, np.concatenate((polylines1, new_polylines2))

def embed(background, image, polylines):
    background = background.copy()

    bgh, bgw, _ = background.shape
    ih, iw, _ = image.shape

    yoff = round((bgh - ih) / 2)
    xoff = round((bgw - iw) / 2)

    background[yoff:yoff + ih, xoff:xoff + iw] = image

    polylines = polylines.copy()
    polylines[:,2::2] += yoff
    polylines[:,1::2] += xoff

    return background, polylines

def crop(image, polylines):
    polylines = polylines.copy()
    polylines[:,:, 1] -= 500
    polylines[:,:, 0] -= 500
    return image[500:1500, 500:1500],  polylines

def augment_demo():
    image, polylines = read_data('./dataset', 'age1_3p_altar')

    for i, image, polylines in augment(image, polylines, 10):
        write_data(image, polylines, 'stuff', 'train', f'age1_3p_altar_{i:03}')

def stack_demo():
    image1, polylines1 = read_data('./dataset', 'age1_3p_altar')
    image2, polylines2 = read_data('./dataset', 'age1_3p_baths')
    image3, polylines3 = read_data('./dataset', 'age1_3p_theater')
    image4, polylines4 = read_data('./dataset', 'age1_4p_well')
    image5, polylines5 = read_data('./dataset', 'age1_5p_altar')
    image6, polylines6 = read_data('./dataset', 'age1_6p_theater')
    image7, polylines7 = read_data('./dataset', 'age1_7p_baths')
    image8, polylines8 = read_data('./dataset', 'age1_7p_well')

    image, polylines = stack(image1, polylines1, image2, polylines2)
    image, polylines = stack(image, polylines, image3, polylines3)
    image, polylines = stack(image, polylines, image4, polylines4)
    image, polylines = stack(image, polylines, image5, polylines5)
    image, polylines = stack(image, polylines, image6, polylines6)
    image, polylines = stack(image, polylines, image7, polylines7)
    image, polylines = stack(image, polylines, image8, polylines8)
    
    write_data(image, polylines, 'new_dataset', 'test')

def combo_demo():
    image1, polylines1 = read_data('./dataset', 'age1_3p_altar')
    image2, polylines2 = read_data('./dataset', 'age1_3p_baths')
    image3, polylines3 = read_data('./dataset', 'age1_3p_theater')

    image, polylines = stack(image1, polylines1, image2, polylines2)
    image, polylines = stack(image, polylines, image3, polylines3)

    for i, image, polylines in augment(image, polylines, 10):
        write_data(image, polylines, 'stuff2', 'train', f'combo_{i:03}')

def stuff():
    blue_cards = [
        read_data('./dataset', 'age1_3p_altar'),
        read_data('./dataset', 'age1_3p_baths'),
        read_data('./dataset', 'age1_3p_theater'),
        read_data('./dataset', 'age1_4p_well'),
        read_data('./dataset', 'age1_5p_altar'),
        read_data('./dataset', 'age1_6p_theater'),
        read_data('./dataset', 'age1_7p_baths'),
        read_data('./dataset', 'age1_7p_well'),
        read_data('./dataset', 'age2_3p_aqueduct'),
        read_data('./dataset', 'age2_3p_courthouse'),
        read_data('./dataset', 'age2_3p_statue'),
        read_data('./dataset', 'age2_3p_temple'),
        read_data('./dataset', 'age2_5p_courthouse'),
        read_data('./dataset', 'age2_6p_temple'),
        read_data('./dataset', 'age2_7p_aqueduct'),
        read_data('./dataset', 'age2_7p_statue'),
        read_data('./dataset', 'age3_3p_gardens'),
        read_data('./dataset', 'age3_3p_palace'),
        read_data('./dataset', 'age3_3p_pantheon'),
        read_data('./dataset', 'age3_3p_senate'),
        read_data('./dataset', 'age3_3p_town_hall'),
        read_data('./dataset', 'age3_4p_gardens'),
        read_data('./dataset', 'age3_5p_senate'),
        read_data('./dataset', 'age3_6p_pantheon'),
        read_data('./dataset', 'age3_6p_town_hall'),
        read_data('./dataset', 'age3_7p_palace'),
    ]

    green_cards = [
        read_data('./dataset', 'age1_3p_apothecary'),
        read_data('./dataset', 'age1_3p_scriptorium'),
        read_data('./dataset', 'age1_3p_workshop'),
        read_data('./dataset', 'age1_4p_scriptorium'),
        read_data('./dataset', 'age1_5p_apothecary'),
        read_data('./dataset', 'age1_7p_workshop'),
        read_data('./dataset', 'age2_3p_dispensary'),
        read_data('./dataset', 'age2_3p_laboratory'),
        read_data('./dataset', 'age2_3p_library'),
        read_data('./dataset', 'age2_3p_school'),
        read_data('./dataset', 'age2_4p_dispensary'),
        read_data('./dataset', 'age2_5p_laboratory'),
        read_data('./dataset', 'age2_6p_library'),
        read_data('./dataset', 'age2_7p_school'),
        read_data('./dataset', 'age3_3p_academy'),
        read_data('./dataset', 'age3_3p_lodge'),
        read_data('./dataset', 'age3_3p_observatory'),
        read_data('./dataset', 'age3_3p_study'),
        read_data('./dataset', 'age3_3p_university'),
        read_data('./dataset', 'age3_4p_university'),
        read_data('./dataset', 'age3_5p_study'),
        read_data('./dataset', 'age3_6p_lodge'),
        read_data('./dataset', 'age3_7p_academy'),
        read_data('./dataset', 'age3_7p_obsersvatory'),
    ]

    for k in range(20):
        cards = random.sample(green_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 100):
            write_data(image, polylines, 'new_dataset_green', 'train', f'train_{(k * 100 + i):04}')

    for k in range(20):
        cards = random.sample(green_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 5):
            write_data(image, polylines, 'new_dataset_green', 'val', f'val_{(k * 5 + i):04}')

stuff()
