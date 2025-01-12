from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.detection_utils import annotations_to_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

import numpy, os, cv2, torch, random

from albumentations import Compose, ShiftScaleRotate, RandomBrightnessContrast, KeypointParams

from utils.database import all_cards
from utils.augment import resize, stack, embed

def generated_mapper(dataset_dict):
    cards = [all_cards[dataset_dict["image_id"] % len(all_cards)]] + random.sample(all_cards, random.randint(1, 5))
    
    image, polylines = cards[0]
    for img, pol in cards[1:]:
        image, polylines = stack(image, polylines, img, pol)

    old_height, old_width = image.shape[:2]
    image, polylines = resize(image, polylines, 400, (int)(old_width * (400 / old_height)))

    background = cv2.imread(f'chequered_0215.jpg')
    background = cv2.resize(background, (640, 640))
    image, polylines = embed(background, image, polylines)

    augmentation = Compose(
        [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.8, 0.5), rotate_limit=180, p=1, border_mode=cv2.BORDER_CONSTANT),
#            RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.4),
        ],
        keypoint_params=KeypointParams(format='xy', remove_invisible=False)
    )
    
    augmented = augmentation(image=image, keypoints=polylines[:,1:].reshape(-1, 2))
    image = augmented['image']
    polylines[:,1:] = augmented['keypoints'].reshape(-1, 8)

    # os.makedirs("stuff", exist_ok=True)
    # image = cv2.polylines(image, [numpy.array(polyline[1:].reshape(-1, 2), dtype=numpy.int32) for polyline in polylines], isClosed=True, color=(255, 0, 255), thickness=4)
    # cv2.imwrite(f'stuff/{dataset_dict["image_id"]}.jpg', image)

    return {
        "image": torch.from_numpy(image.transpose(2, 0, 1)),
        "width": 640,
        "height": 640,
        "instances": annotations_to_instances([
            {
                "bbox": [numpy.min(polyline[1::2]), numpy.min(polyline[2::2]), numpy.max(polyline[1::2]), numpy.max(polyline[2::2])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [polyline[1:].tolist()],
                "category_id": polyline[0],
            } for polyline in polylines
        ], image.shape[1:])
    }

class GeneratedTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(GeneratedTrainer, self).__init__(cfg)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=generated_mapper)

def main():
    DatasetCatalog.register("cards_train", lambda: [{"image_id": i, "height": 640, "width": 640} for i in range(20000)])
    MetadataCatalog.get("cards_train").set(thing_classes=["card", "military_power", "blue_card", "green_card_tablet", "green_card_compass", "green_card_cog", "blue_card_3", "blue_card_4", "blue_card_5", "blue_card_6", "blue_card_7", "blue_card_8", "green_card"])
    DatasetCatalog.register("cards_val", lambda: [{"image_id": i, "height": 640, "width": 640} for i in range(1000)])
    MetadataCatalog.get("cards_val").set(thing_classes=["card", "military_power", "blue_card", "green_card_tablet", "green_card_compass", "green_card_cog", "blue_card_3", "blue_card_4", "blue_card_5", "blue_card_6", "blue_card_7", "blue_card_8", "green_card"])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cards_train","cards_val")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = GeneratedTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    print(cfg)
    return trainer.train()

if __name__ == "__main__":
    main()
