# yoro

You only run once to generate train set

# why

Just want to make a easy way to generate yolov3 train set and cfg automatically

# usage

```bash

1、 git clone https://github.com/leegohi/yoro.git

2、 cd yoro

3、 python yolo.py --name 2019 --img ./imges_folder --voc ./voc_folder --cls ./2019.names --prt 0.8

```

# output

```python
.
├── 2019.names
├── LICENSE
├── README.md
├── TrainSet-2019
│   ├── 2019.names
│   ├── backup
│   ├── exclude
│   ├── train.data
│   ├── train.txt
│   ├── train_test.txt
│   ├── voc
│   ├── yolo
│   └── yolov3.cfg
├── train.data.template
├── yolov3.cfg.template
└── yoro.py
```

after these commands,you will see like below,the script generate a folder names **TrainSet-2019** contains:

- **2019.names** ：copy from ./2019.names contains the classes，it is just a txt file，each line put one label.

- **backup** ：when training the darknet put it weight file to it

- **exclude** ： useless image，ignore

- **train.data** : the file tell darknet where the train image,test image,classes count,classes file,backup folder are

- **voc** : collect the .xml voc file

- **yolo**: collect the train images and corresponding yolo txt format file

- **yolov3.cfg** : yolov3.cfg,the script just replace the classes and filters

# command

```python
usage: yoro.py [-h] -n NAME -i IMG -v VOC -c CLS [-p PRT]

optional arguments:
-h, --help show this help message and exit
-n NAME, --name NAME name of trainset
-i IMG, --img IMG path to image
-v VOC, --voc VOC path to voc
-c CLS, --cls CLS path to classes file
-p PRT, --prt PRT how much percentage of image will be used to train.
e.g. if you has labeled 100 images you want to use 80
for train and 20 for test,so the percentage is 0.8
```

# Train

now you need get the pretrain file:

```bash
wget https://pjreddie.com/media/files/darknet53.conv.74

```

**start training**

```bash

./darknet detector train  TrainSet-2019/train.data   TrainSet-2019/yolov3.cfg darknet53.conv.74

```

remind that you need change the path to yourself above
