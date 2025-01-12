import cv2, random, os, shutil

from utils.database import all_cards, write_card
from utils.augment import augment, stack, embed, resize

def generate_new_database(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(f'{folder}/images/train')
    os.makedirs(f'{folder}/labels/train')
    os.makedirs(f'{folder}/images/val')
    os.makedirs(f'{folder}/labels/val')

    background = cv2.imread(f'chequered_0215.jpg')
    background = cv2.resize(background, (640, 640))
    
    for k in range(20000):
        cards = [all_cards[k % len(all_cards)]] + random.sample(all_cards, random.randint(2, 5))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        old_height, old_width = image.shape[:2]
        image, polylines = resize(image, polylines, 400, (int)(old_width * (400 / old_height)))
        image, polylines = embed(background, image, polylines)
        image, polylines = augment(image, polylines)

        write_card(image, polylines, 'new_dataset', 'train', f'train_{k}')
    
    for k in range(5000):
        cards = [all_cards[k % len(all_cards)]] + random.sample(all_cards, random.randint(2, 5))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        old_height, old_width = image.shape[:2]
        image, polylines = resize(image, polylines, 400, (int)(old_width * (400 / old_height)))
        image, polylines = embed(background, image, polylines)
        image, polylines = augment(image, polylines)

        write_card(image, polylines, 'new_dataset', 'val', f'val_{k}')

generate_new_database('new_dataset')
