import cv2, numpy, os

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
    return image, numpy.array([denormalize_keypoints(polyline, image.shape) for polyline in numpy.loadtxt(f'{basepath}/labels/{type}/{filename}.txt').reshape(-1, 9) if polyline[0] in [2, 12, 13, 14, 15, 16, 17]])

def write_card(image, polylines, basepath, type, filename):
    # image = cv2.polylines(image, [numpy.array(polyline[1:].reshape(-1, 2), dtype=numpy.int32) for polyline in polylines], isClosed=True, color=(255, 0, 255), thickness=4)
    cv2.imwrite(f'{basepath}/images/{type}/{filename}.jpg', image)
    numpy.savetxt(f'{basepath}/labels/{type}/{filename}.txt', [normalize_keypoints(polyline, image.shape) for polyline in polylines.reshape(-1, 9)], '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f')

def read_backgrounds(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(subdir, file)
            background = cv2.imread(path)
            background = cv2.resize(background, (640, 640))
            yield background
        for dir in dirs:
            path = os.path.join(subdir, dir)
            read_backgrounds(path)

backgrounds = list(read_backgrounds('./dataset/backgrounds'))

brown_cards = [
    read_card('./dataset', 'age1_3p_clay_pit'),
    read_card('./dataset', 'age1_3p_clay_pool'),
    read_card('./dataset', 'age1_3p_lumber_yard'),
    read_card('./dataset', 'age1_3p_ore_vein'),
    read_card('./dataset', 'age1_3p_stone_pit'),
    read_card('./dataset', 'age1_3p_timber_yard'),
    read_card('./dataset', 'age1_4p_excavation'),
    read_card('./dataset', 'age1_4p_lumber_yard'),
    read_card('./dataset', 'age1_4p_ore_vein'),
    read_card('./dataset', 'age1_5p_clay_pool'),
    read_card('./dataset', 'age1_5p_forest_cave'),
    read_card('./dataset', 'age1_5p_stone_pit'),
    read_card('./dataset', 'age1_6p_mine'),
    read_card('./dataset', 'age1_6p_tree_farm'),
    read_card('./dataset', 'age2_3p_brickyard'),
    read_card('./dataset', 'age2_3p_foundry'),
    read_card('./dataset', 'age2_3p_quarry'),
    read_card('./dataset', 'age2_3p_sawmill'),
    read_card('./dataset', 'age2_4p_brickyard'),
    read_card('./dataset', 'age2_4p_foundry'),
    read_card('./dataset', 'age2_4p_quarry'),
    read_card('./dataset', 'age2_4p_sawmill'),
]

red_cards = [
    read_card('./dataset', 'age1_3p_barracks'),
    read_card('./dataset', 'age1_3p_guard_tower'),
    read_card('./dataset', 'age1_3p_stockade'),
    read_card('./dataset', 'age1_4p_guard_tower'),
    read_card('./dataset', 'age1_5p_barracks'),
    read_card('./dataset', 'age1_7p_stockade'),
    read_card('./dataset', 'age2_3p_archery_range'),
    read_card('./dataset', 'age2_3p_stables'),
    read_card('./dataset', 'age2_3p_walls'),
    read_card('./dataset', 'age2_4p_training_ground'),
    read_card('./dataset', 'age2_5p_stables'),
    read_card('./dataset', 'age2_6p_archery_range'),
    read_card('./dataset', 'age2_6p_training_ground'),
    read_card('./dataset', 'age2_7p_training_ground'),
    read_card('./dataset', 'age2_7p_walls'),
    read_card('./dataset', 'age3_3p_arsenal'),
    read_card('./dataset', 'age3_3p_fortifications'),
    read_card('./dataset', 'age3_3p_siege_workshop'),
    read_card('./dataset', 'age3_4p_castrum'),
    read_card('./dataset', 'age3_4p_circus'),
    read_card('./dataset', 'age3_5p_arsenal'),
    read_card('./dataset', 'age3_5p_siege_workshop'),
    read_card('./dataset', 'age3_6p_circus'),
    read_card('./dataset', 'age3_7p_castrum'),
    read_card('./dataset', 'age3_7p_fortifications'),
]

yellow_cards = [
    read_card('./dataset', 'age1_3p_east_trading_post'),
    read_card('./dataset', 'age1_3p_marketplace'),
    read_card('./dataset', 'age1_3p_west_trading_post'),
    read_card('./dataset', 'age1_4p_tavern'),
    read_card('./dataset', 'age1_5p_tavern'),
    read_card('./dataset', 'age1_6p_marketplace'),
    read_card('./dataset', 'age1_7p_east_trading_post'),
    read_card('./dataset', 'age1_7p_tavern'),
    read_card('./dataset', 'age1_7p_west_trading_post'),
    read_card('./dataset', 'age2_3p_caravansery'),
    read_card('./dataset', 'age2_3p_forum'),
    read_card('./dataset', 'age2_3p_vineyard'),
    read_card('./dataset', 'age2_4p_bazar'),
    read_card('./dataset', 'age2_5p_caravansery'),
    read_card('./dataset', 'age2_6p_caravansery'),
    read_card('./dataset', 'age2_6p_forum'),
    read_card('./dataset', 'age2_6p_vineyard'),
    read_card('./dataset', 'age2_7p_bazar'),
    read_card('./dataset', 'age2_7p_forum'),
    read_card('./dataset', 'age3_3p_arena'),
    read_card('./dataset', 'age3_3p_haven'),
    read_card('./dataset', 'age3_3p_lighthouse'),
    read_card('./dataset', 'age3_4p_chamber_of_commerce'),
    read_card('./dataset', 'age3_4p_haven'),
    read_card('./dataset', 'age3_5p_arena'),
    read_card('./dataset', 'age3_5p_ludus'),
    read_card('./dataset', 'age3_6p_chamber_of_commerce'),
    read_card('./dataset', 'age3_6p_lighthouse'),
    read_card('./dataset', 'age3_7p_ludus'),
]

purple_cards = [
    read_card('./dataset', 'age3_3p_builders_guild'),
    read_card('./dataset', 'age3_3p_craftsmens_guild'),
    read_card('./dataset', 'age3_3p_decorators_guild'),
    read_card('./dataset', 'age3_3p_magistrates_guild'),
    read_card('./dataset', 'age3_3p_philosophers_guild'),
    read_card('./dataset', 'age3_3p_scientists_guild'),
    read_card('./dataset', 'age3_3p_shipowners_guild'),
    read_card('./dataset', 'age3_3p_spies_guild'),
    read_card('./dataset', 'age3_3p_traders_guild'),
    read_card('./dataset', 'age3_3p_workers_guild'),
]

grey_cards = [
    read_card('./dataset', 'age1_3p_glassworks'),
    read_card('./dataset', 'age1_3p_loom'),
    read_card('./dataset', 'age1_3p_press'),
    read_card('./dataset', 'age1_6p_glassworks'),
    read_card('./dataset', 'age1_6p_loom'),
    read_card('./dataset', 'age1_6p_press'),
    read_card('./dataset', 'age2_3p_glassworks'),
    read_card('./dataset', 'age2_3p_loom'),
    read_card('./dataset', 'age2_3p_press'),
    read_card('./dataset', 'age2_5p_glassworks'),
    read_card('./dataset', 'age2_5p_loom'),
    read_card('./dataset', 'age2_5p_press'),
]

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

all_cards = brown_cards + grey_cards + red_cards + yellow_cards + blue_cards + green_cards + purple_cards

classes = [
    "card",
    "military_power",
    "blue_card",
    "green_card_tablet",
    "green_card_compass",
    "green_card_cog",
    "blue_card_3",
    "blue_card_4",
    "blue_card_5",
    "blue_card_6",
    "blue_card_7",
    "blue_card_8",
    "green_card",
    "yellow_card",
    "grey_card",
    "brown_card",
    "red_card",
    "purple_card"
]
