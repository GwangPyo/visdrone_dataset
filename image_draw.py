from matplotlib.colors import to_rgb
import cv2

color_maps = ['grey', 'r', 'b', 'g', 'gold', 'm', 'c', 'orange', 'olive', 'purple', 'pink', 'k']
color_maps = list(map(to_rgb, color_maps))

for i, c in enumerate(color_maps):
    r, g, b = c
    c = (b, g, r)
    color_maps[i] = tuple(map(lambda x: int(x * 255), c))


def draw(image, annotation):
    im_show = image.copy()
    # Display the image
    if type(annotation) is list:
        bbox, label, _ = annotation
        # Create a Rectangle patch
    elif type(annotation) is dict:
        bbox = annotation["boxes"]
        label = annotation["labels"]
    else:
        raise TypeError("annotation type must be list or dict")
    for bbox, class_label in zip(bbox, label):
        # bbox := [x1, y1, x2, y2]
        [x1, y1, x2, y2] = bbox
        cv2.rectangle(im_show, (x1, y1), (x2, y2), thickness=1, color=color_maps[class_label])
    cv2.imshow("image", im_show)
    cv2.waitKey(0)
    return im_show