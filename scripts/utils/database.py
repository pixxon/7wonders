import cv2, numpy

def normalize_keypoints(keypoints: numpy.ndarray, image_shape: tuple[int, int]):
    height, width = image_shape[:2]
    normalized = keypoints.copy()
    normalized[1::2] /= width
    normalized[2::2] /= height
    return normalized

def denormalize_keypoints(keypoints: numpy.ndarray, image_shape: tuple[int, int]):
    height, width = image_shape[:2]
    denormalized = keypoints.copy()
    denormalized[1::2] *= width
    denormalized[2::2] *= height
    return denormalized

def read_card(basepath, filename, type='train'):
    image = cv2.imread(f'{basepath}/images/{type}/{filename}.jpg')
    resize_ratio = (400 / float(image.shape[0]))
    image = cv2.resize(image, None, fx = resize_ratio, fy = resize_ratio)
    return image, numpy.array([denormalize_keypoints(polyline, image.shape) for polyline in numpy.loadtxt(f'{basepath}/labels/{type}/{filename}.txt').reshape(-1, 9) if polyline[0] in [12, 2]])

def write_card(image, polylines, basepath, type, filename):
    image = cv2.polylines(image, [numpy.array(polyline[1:].reshape(-1, 2), dtype=numpy.int32) for polyline in polylines], isClosed=True, color=(255, 0, 255), thickness=4)
    cv2.imwrite(f'{basepath}/images/{type}/{filename}.jpg', image)
    numpy.savetxt(f'{basepath}/labels/{type}/{filename}.txt', [normalize_keypoints(polyline, image.shape) for polyline in polylines.reshape(-1, 9)], '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f')

blue_cards = [
    read_card('./dataset', 'age1_3p_altar'),
    read_card('./dataset', 'age1_3p_baths'),
    read_card('./dataset', 'age1_3p_theater'),
    read_card('./dataset', 'age1_4p_well'),
    read_card('./dataset', 'age1_5p_altar'),
    read_card('./dataset', 'age1_6p_theater'),
    read_card('./dataset', 'age1_7p_baths'),
    read_card('./dataset', 'age1_7p_well'),
    read_card('./dataset', 'age2_3p_aqueduct'),
    read_card('./dataset', 'age2_3p_courthouse'),
    read_card('./dataset', 'age2_3p_statue'),
    read_card('./dataset', 'age2_3p_temple'),
    read_card('./dataset', 'age2_5p_courthouse'),
    read_card('./dataset', 'age2_6p_temple'),
    read_card('./dataset', 'age2_7p_aqueduct'),
    read_card('./dataset', 'age2_7p_statue'),
    read_card('./dataset', 'age3_3p_gardens'),
    read_card('./dataset', 'age3_3p_palace'),
    read_card('./dataset', 'age3_3p_pantheon'),
    read_card('./dataset', 'age3_3p_senate'),
    read_card('./dataset', 'age3_3p_town_hall'),
    read_card('./dataset', 'age3_4p_gardens'),
    read_card('./dataset', 'age3_5p_senate'),
    read_card('./dataset', 'age3_6p_pantheon'),
    read_card('./dataset', 'age3_6p_town_hall'),
    read_card('./dataset', 'age3_7p_palace'),
]

green_cards = [
    read_card('./dataset', 'age1_3p_apothecary'),
    read_card('./dataset', 'age1_3p_scriptorium'),
    read_card('./dataset', 'age1_3p_workshop'),
    read_card('./dataset', 'age1_4p_scriptorium'),
    read_card('./dataset', 'age1_5p_apothecary'),
    read_card('./dataset', 'age1_7p_workshop'),
    read_card('./dataset', 'age2_3p_dispensary'),
    read_card('./dataset', 'age2_3p_laboratory'),
    read_card('./dataset', 'age2_3p_library'),
    read_card('./dataset', 'age2_3p_school'),
    read_card('./dataset', 'age2_4p_dispensary'),
    read_card('./dataset', 'age2_5p_laboratory'),
    read_card('./dataset', 'age2_6p_library'),
    read_card('./dataset', 'age2_7p_school'),
    read_card('./dataset', 'age3_3p_academy'),
    read_card('./dataset', 'age3_3p_lodge'),
    read_card('./dataset', 'age3_3p_observatory'),
    read_card('./dataset', 'age3_3p_study'),
    read_card('./dataset', 'age3_3p_university'),
    read_card('./dataset', 'age3_4p_university'),
    read_card('./dataset', 'age3_5p_study'),
    read_card('./dataset', 'age3_6p_lodge'),
    read_card('./dataset', 'age3_7p_academy'),
    read_card('./dataset', 'age3_7p_obsersvatory'),
]

all_cards = blue_cards + green_cards
