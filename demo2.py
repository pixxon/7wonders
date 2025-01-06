# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg

import numpy as np
import os, json, cv2

def denormalize_keypoints(keypoints: np.ndarray, image_shape: tuple[int, int]):
    height, width = image_shape[:2]
    denormalized = keypoints.copy()
    denormalized[1::2] *= width
    denormalized[2::2] *= height
    return denormalized

def read_data(basepath, folder, filename):
    image = cv2.imread(f'{basepath}/images/{folder}/{filename}.jpg')
    polylines = [denormalize_keypoints(polyline, image.shape) for polyline in np.loadtxt(f'{basepath}/labels/{folder}/{filename}.txt').reshape(-1, 9)]
    return {
        "file_name": f'{basepath}/images/{folder}/{filename}.jpg',
        # "image": image,
        "image_id": filename,
        "height": image.shape[0],
        "width": image.shape[1],
        "annotations": [
            {
                "bbox": [np.min(polyline[1::2]), np.min(polyline[2::2]), np.max(polyline[1::2]), np.max(polyline[2::2])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [polyline[1:].tolist()],
                "category_id": polyline[0],
            } for polyline in polylines
        ]
    }

def get_cards():
    return [
        read_data('./dataset', 'train', 'age1_3p_altar'),
        read_data('./dataset', 'train', 'age1_3p_baths'),
        read_data('./dataset', 'train', 'age1_3p_theater'),
        read_data('./dataset', 'train', 'age1_4p_well'),
        read_data('./dataset', 'train', 'age1_5p_altar'),
        read_data('./dataset', 'train', 'age1_6p_theater'),
        read_data('./dataset', 'train', 'age1_7p_baths'),
        read_data('./dataset', 'train', 'age1_7p_well'),
        read_data('./dataset', 'train', 'age2_3p_aqueduct'),
        read_data('./dataset', 'train', 'age2_3p_courthouse'),
        read_data('./dataset', 'train', 'age2_3p_statue'),
        read_data('./dataset', 'train', 'age2_3p_temple'),
        read_data('./dataset', 'train', 'age2_5p_courthouse'),
        read_data('./dataset', 'train', 'age2_6p_temple'),
        read_data('./dataset', 'train', 'age2_7p_aqueduct'),
        read_data('./dataset', 'train', 'age2_7p_statue'),
        read_data('./dataset', 'train', 'age3_3p_gardens'),
        read_data('./dataset', 'train', 'age3_3p_palace'),
        read_data('./dataset', 'train', 'age3_3p_pantheon'),
        read_data('./dataset', 'train', 'age3_3p_senate'),
        read_data('./dataset', 'train', 'age3_3p_town_hall'),
        read_data('./dataset', 'train', 'age3_4p_gardens'),
        read_data('./dataset', 'train', 'age3_5p_senate'),
        read_data('./dataset', 'train', 'age3_6p_pantheon'),
        read_data('./dataset', 'train', 'age3_6p_town_hall'),
        read_data('./dataset', 'train', 'age3_7p_palace'),
    ]

def get_cards_train():
    return [read_data('./new_dataset_mixed', 'train', f'train_{i:04}') for i in range(2000)]

def get_cards_val():
    return [read_data('./new_dataset_mixed', 'val', f'val_{i:04}') for i in range(100)]

def main():
    DatasetCatalog.register("cards_train", lambda: get_cards_train())
    MetadataCatalog.get("cards_train").set(thing_classes=["victory_point", "military_power", "blue_card", "green_card_tablet", "green_card_compass", "green_card_cog", "blue_card_3", "blue_card_4", "blue_card_5", "blue_card_6", "blue_card_7", "blue_card_8", "green_card"])
    DatasetCatalog.register("cards_val", lambda: get_cards_val())
    MetadataCatalog.get("cards_val").set(thing_classes=["victory_point", "military_power", "blue_card", "green_card_tablet", "green_card_compass", "green_card_cog", "blue_card_3", "blue_card_4", "blue_card_5", "blue_card_6", "blue_card_7", "blue_card_8", "green_card"])

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cards_train","cards_val")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    return trainer.train()

import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm

def _find_free_port():
    """
    Copied from detectron2/engine/launch.py
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, nprocs, args=()):
    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    # dist_url = "env://"
    mp.spawn(
        distributed_worker, nprocs=nprocs, args=(main_func, nprocs, dist_url, args), daemon=False
    )


def distributed_worker(local_rank, main_func, nprocs, dist_url, args):
    dist.init_process_group(
        backend="gloo", init_method=dist_url, world_size=nprocs, rank=local_rank
    )
    comm.synchronize()
    assert comm._LOCAL_PROCESS_GROUP is None
    pg = dist.new_group(list(range(nprocs)))
    comm._LOCAL_PROCESS_GROUP = pg
    main_func(*args)

def invoke_main() -> None:
    launch(main, 1)

if __name__ == "__main__":
    invoke_main()  # pragma: no cover
