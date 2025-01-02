train:
```
yolo segment train data=../new_dataset/data.yaml model=yolo11n-seg.yaml
```

predict:
```
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg imgsz=2016
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg imgsz=2016 conf=0.9
```
