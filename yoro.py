from __future__ import print_function
import os
from pathlib import Path
from random import shuffle
import xml.etree.ElementTree as ET
import shutil
import argparse


class AutoYolo():

    def __init__(self, prefix, class_file, voc_path, img_path, train_percentage=0.8):
        self.train_set = Path("TrainSet-%s" % prefix)
        self.local_data = self.train_set/"yolo"
        self.local_exclude = self.train_set/"exclude"
        self.local_annos = self.train_set/"voc"
        self.local_backup = self.train_set/"backup"
        self.voc_path = Path(voc_path)
        self.img_path = Path(img_path)
        self.class_file = Path(class_file)
        self.percentage = train_percentage
        if not self.train_set.exists():
            print("mkdirs", self.local_data,
                  self.local_exclude, self.local_annos)
            self.train_set.mkdir()
            self.local_data.mkdir()
            self.local_exclude.mkdir()
            self.local_annos.mkdir()
            self.local_backup.mkdir()
        self.load_classes()

    def load_classes(self):
        with open(self.class_file, "r") as f:
            self.classes = f.read().split("\n")
        shutil.copy(self.class_file, self.train_set)
        self.class_file = self.train_set/self.class_file.name

    def configure(self):
        for x in self.img_path.iterdir():
            img_data = x
            xml_data = self.voc_path / x.with_suffix(".xml").name
            if not xml_data.exists():
                continue
            xml_f_size = xml_data.stat().st_size
            if xml_f_size >= 500:
                shutil.copy(img_data, self.local_data)
                shutil.copy(xml_data, self.local_annos)
                continue
            shutil.copy(img_data, self.local_exclude)
        base = self.local_data
        imgs = list(map(lambda p: p.resolve().as_posix(), base.iterdir()))
        total = len(imgs)
        print("total label images:", total)
        print("shuffle label imgs")
        shuffle(imgs)
        split_index = int(total*self.percentage)
        print("train label image:", split_index)
        print("test label image:", total-split_index)
        self.train_txt = self.train_set/"train.txt"
        self.train_test_txt = self.train_set/"train_test.txt"
        with open(self.train_txt, 'w') as fw, open(self.train_test_txt, 'w') as ft:
            fw.write("\n".join(imgs[:split_index]))
            ft.write("\n".join(imgs[split_index:]))

        def convert(tx):
            image_ids = open(tx).read().strip().split()
            for image_id in image_ids:
                self.convert_annotation(image_id)
        convert(self.train_txt)
        convert(self.train_test_txt)

    def convert_annotation(self, image_id):
        def convert(size, box):
            dw = 1./(size[0])
            dh = 1./(size[1])
            x = (box[0] + box[1])/2.0 - 1
            y = (box[2] + box[3])/2.0 - 1
            w = box[1] - box[0]
            h = box[3] - box[2]
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            return (x, y, w, h)
        in_file = open(self.local_annos /
                       Path(image_id).with_suffix(".xml").name)
        out_file = open(self.local_data /
                        Path(image_id).with_suffix(".txt").name, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')

    def configure_template(self):
        class_count = len(self.classes)
        with open("train.data.template", "r") as f:
            train_data = f.read().format(classes=class_count,
                                         train=self.train_txt.resolve(),
                                         valid=self.train_test_txt.resolve(),
                                         names=self.class_file.resolve(),
                                         backup=self.local_backup.resolve())
            with (self.train_set/"train.data").open(mode="w") as ft:
                ft.write(train_data)
        with open("yolov3.cfg.template", "r") as f:
            cfg = f.read().format(classes=class_count, filters=(class_count+5)*3)
            with (self.train_set/"yolov3.cfg").open(mode="w") as ft:
                ft.write(cfg)


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True,
                    help="name of trainset")
    ap.add_argument("-i", "--img", required=True,
                    help="path to image")
    ap.add_argument("-v", "--voc", required=True,
                    help="path to voc ")
    ap.add_argument("-c", "--cls", required=True,
                    help="path to classes file ")
    ap.add_argument("-p", "--prt", type=float, default=0.8,
                    help="how much percentage of image will be used to train. e.g. if you has labeled 100 images you want to use 80 \
                    for train and 20 for test,so the percentage is 0.8")
    args = vars(ap.parse_args())

    au = AutoYolo(prefix=args["name"], class_file=args["cls"], voc_path=args["voc"],
                  img_path=args["img"], train_percentage=args["prt"])
    au.configure()
    au.configure_template()


if __name__ == "__main__":
    main()
