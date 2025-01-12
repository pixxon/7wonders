from .utils.database import blue_cards, write_card
from .utils.augment import augment, stack

def augment_demo():
    image, polylines = blue_cards[0]

    for i, image, polylines in augment(image, polylines, 10):
        write_card(image, polylines, 'stuff', 'train', f'age1_3p_altar_{i:03}')

def stack_demo():
    image1, polylines1 = blue_cards[0]
    image2, polylines2 = blue_cards[1]
    image3, polylines3 = blue_cards[2]
    image4, polylines4 = blue_cards[3]
    image5, polylines5 = blue_cards[4]
    image6, polylines6 = blue_cards[5]
    image7, polylines7 = blue_cards[6]
    image8, polylines8 = blue_cards[7]

    image, polylines = stack(image1, polylines1, image2, polylines2)
    image, polylines = stack(image, polylines, image3, polylines3)
    image, polylines = stack(image, polylines, image4, polylines4)
    image, polylines = stack(image, polylines, image5, polylines5)
    image, polylines = stack(image, polylines, image6, polylines6)
    image, polylines = stack(image, polylines, image7, polylines7)
    image, polylines = stack(image, polylines, image8, polylines8)
    
    write_card(image, polylines, 'new_dataset', 'test')

def combo_demo():
    image1, polylines1 = blue_cards[0]
    image2, polylines2 = blue_cards[1]
    image3, polylines3 = blue_cards[2]

    image, polylines = stack(image1, polylines1, image2, polylines2)
    image, polylines = stack(image, polylines, image3, polylines3)

    for i, image, polylines in augment(image, polylines, 10):
        write_card(image, polylines, 'stuff2', 'train', f'combo_{i:03}')
