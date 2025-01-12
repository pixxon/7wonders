import cv2
import random

from .utils.database import green_cards, blue_cards, write_card
from .utils.augment import augment, stack, embed

def generate_new_database(folder):
    for k in range(20):
        cards = random.sample(green_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 100):
            write_card(image, polylines, folder, 'train', f'train_{(k * 100 + i):04}')

    for k in range(20):
        cards = random.sample(green_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 5):
            write_card(image, polylines, folder, 'val', f'val_{(k * 5 + i):04}')
    
    for k in range(20):
        cards = random.sample(blue_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 100):
            write_card(image, polylines, folder, 'train', f'train_{(k * 100 + i + 2000):04}')

    for k in range(20):
        cards = random.sample(blue_cards, random.randint(3, 6))
        image, polylines = cards[0]
        for img, pol in cards[1:]:
            image, polylines = stack(image, polylines, img, pol)

        background = cv2.imread(f'chequered_0215.jpg')
        background = cv2.resize(background, (2000, 2000))

        image, polylines = embed(background, image, polylines)
        for i, image, polylines in augment(image, polylines, 5):
            write_card(image, polylines, folder, 'val', f'val_{(k * 5 + i + 100):04}')

generate_new_database('new_database')
