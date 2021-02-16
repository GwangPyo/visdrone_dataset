import os
import numpy as np
import cv2
import torch


"""
ignored regions (0), 
pedestrian (1), 
people (2), 
bicycle (3), 
car (4), 
van (5), 
truck (6), 
tricycle (7), 
awning-tricycle (8), 
bus (9), 
motor (10), 
others (11)
"""

ClassMap = {0: "ignored region", 1: "pedestrian", 2: "people",
            3: "bicycle", 4: "car", 5: "van", 6: "truck",
            7: "tricycle", 8: "awning-tricycle", 9: "bus",
            10: "motor", 11: "others"
            }


class VisDroneDataBase(object):
    def __init__(self, path, ignore_0=True):
        self.path = path
        self.image_path = self.path + "/image/"
        self.annotation_path = self.path + "/annotation/"
        self.prefix = os.listdir(self.image_path)
        for i, s in enumerate(self.prefix):
            self.prefix[i] = s.replace(".jpg", ".{}")
        self.ignore_0 = ignore_0

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, index):
        prefix = self.prefix[index]
        image_path = self.image_path + prefix.format("jpg")
        annotation_path = self.annotation_path + prefix.format("txt")
        annotation = VisDroneDataBase.read_annotation(annotation_path)
        image = cv2.imread(image_path)
        return image, annotation

    @staticmethod
    def read_annotation(path, ignore_0=True):
        with open(path, "r") as f:
            annotation_raw = f.readlines()
        bboxes = []
        labels = []
        infos = []
        for line in annotation_raw:
            # remove new line characters
            line = line.replace("\n", "")
            # parse and append
            bbox, label, info = VisDroneDataBase.parse_annotation_line(line)

            if ignore_0 and label == 0:
                continue
            else:
                bboxes.append(bbox)
                labels.append(label)
                infos.append(info)
        return [bboxes, labels, infos]

    @staticmethod
    def parse_annotation_line(line):
        line = line.split(",")
        [bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion] = line
        bbox = line[0:4]
        bbox = list(map(np.float32, bbox))
        # [x, y, w, h] -> [x1, y1, x2, y2]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        infos = [float(score), float(truncation), float(occlusion)]
        return bbox, int(category), infos



class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}



if __name__ == "__main__":
    vdb = VisDroneDataBase(path="visdrone_sample")
    img, annotation = vdb[2]
    print(img.shape)
    img = img.transpose((2, 0, 1))
    print(img.shape)
    import image_draw
    image_draw.draw(img, annotation)