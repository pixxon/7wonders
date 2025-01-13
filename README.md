one-shot download for backgrounds
```
wget https://www.robots.ox.ac.uk/\~vgg/data/dtd/download/dtd-r1.0.1.tar.gz && tar -xf dtd-r1.0.1.tar.gz && rm dtd-r1.0.1.tar.gz && mkdir dataset/backgrounds && mv dtd/images/* dataset/backgrounds/ && rm -r dtd && rm dataset/backgrounds/waffled/.directory
```

train for detecron2 ( trains into the base model name, then uses test.jpg and generates result.jpg )
```
python3 ./scripts/train.py
```

predict only
```
python3 ./scripts/demo3.py
```

train for yolo:
```
yolo segment train data=../new_dataset/data.yaml model=yolo11n-seg.yaml
```

predict:
```
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg imgsz=2016
yolo segment predict model=./runs/segment/train/weights/best.pt source=./test.jpg imgsz=2016 conf=0.9
```

