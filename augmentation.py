from PIL import Image, ImageDraw #version 6.1.0
import PIL #version 1.2.0
import torch
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
from IPython.display import display
import json
from math import floor, ceil
import click

def convert_coco_to_voc(bbox):
    """Fuction converting bounding box from COCO format to VOC format

    Args:
        bbox (list): [x, y, width, height]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    
    return [xmin, ymin, xmax, ymax]


def convert_voc_to_coco(bbox):
    """Fuctions converting bounding box from VOC format to COCO format

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [x, y, width, height]
    """
    x = float(bbox[0])
    y = float(bbox[1])
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    
    return [round(a,2) for a in [ x, y, width, height]]   


def get_img_id(anns, img_name):
    """Finds the image id in annotation, given an image name

    Args:
        anns (dict): COCO annotations
        img_name (string): Image name 

    Raises:
        Exception: When annotations don't contain the image name

    Returns:
        int: Id of the given image 
    """
    images = anns["images"]
    for image in images:
        if image["file_name"] == img_name:
            return image["id"]
        
    raise Exception(f"No image with name {img_name} found")


def parse_coco_annot(annotation_path, image_name):
    """Returns bounding boxes of the given image from annotations

    Args:
        annotation_path (string): Path to the annotation file
        image_name (string): Name of the desired image 

    Returns:
        [dict, dict]: [{"boxes": [boxes in coco format]}, coco annotations]
    """
    f = open(annotation_path, "r")
    anns = json.load(f)
    img_id = get_img_id(anns, image_name)
    
    annotations = anns["annotations"]
    
    boxes = []
    
    for ann in annotations:
        if ann["image_id"] == img_id:
            bbox = ann["bbox"]
            
            boxes.append(convert_coco_to_voc(bbox))
        
    return boxes, anns


def draw_PIL_image(image, boxes, new_path):
    '''
        Draw PIL image
        image: A PIL image
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    if type(boxes) == list:
        for i, box in enumerate(boxes):
            boxes[i] = list(box)
    else:
        boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy= boxes[i], outline=(23,241,123), width = 5)
    
    # display(new_image)
    new_image.save(new_path)




def flip(image, boxes):
    '''
        Flip image horizontally.
        image: a PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
    '''
    new_image = F.hflip(image)
    
    #flip boxes 
    new_boxes = boxes.clone()
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes



def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)
        
        Out: Intersection each of boxes1 with respect to each of boxes2, 
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)

def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes 
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)
        
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of 
             dimensions (n1, n2)
        
        Formula: 
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter) #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union

def filter_out_obscured_boxes(boxes, image_w, image_h):
    """Filters out boxes that aren't entirely in the image

    Args:
        boxes (list): List of boxes in voc format
        image_w (float): Width of the cropped image
        image_h (float): Height of the cropped image

    Returns:
        list: All the boxes that are entirely in the image
    """
    ret = []
    for box in boxes:
        if floor(box[0]) > 0 and floor(box[1]) > 0 and ceil(box[2]) < floor(image_w) and ceil(box[3]) < floor(image_h):
            ret.append(box)
            
    return ret
    
def random_crop(image, boxes):
    '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
    
        Out: cropped image , new boxes
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    
    new_image = image
    new_boxes = boxes
    for _ in range(50):
        # Crop dimensions: [0.3, 1] of original dimensions
        new_h = random.uniform(0.003*original_h, 0.5*original_h)
        new_w = random.uniform(0.003*original_w, 0.5*original_w)
        
        # Aspect ratio constraint b/t .5 & 2
        if new_h/new_w < 0.5 or new_h/new_w > 2:
            continue
        
        #Crop coordinate
        left = random.uniform(0, original_w - new_w)
        right = left + new_w
        top = random.uniform(0, original_h - new_h)
        bottom = top + new_h
        crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])
        
        #Crop
        new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)
        
        #Center of bounding boxes
        center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0
        
        #Find bounding box has been had center in crop
        center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #( #objects)
        
        if not center_in_crop.any():
            continue
        
        #take matching bounding box
        new_boxes = boxes[center_in_crop, :]
        
        #Use the box left and top corner or the crop's
        new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
        
        #adjust to crop
        new_boxes[:, :2] -= crop[:2]
        
        new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])
        
        #adjust to crop
        new_boxes[:, 2:] -= crop[:2]
        new_boxes = filter_out_obscured_boxes(new_boxes, new_w, new_h)
        if len(new_boxes) == 0:     # Cropped image contain no boxes
            continue
        
        return F.to_pil_image(new_image), new_boxes

def get_largest_id(anns):
    '''
        anns: list of annotations to be searched for largest id
        
        Out: the largest id of annotations
    '''
    max_id = float('-inf')
    for ann in anns:
        if ann["id"] > max_id:
            max_id = ann["id"]
    
    return max_id
    
def add_boxes_to_coco(boxes, anns, image, image_path):
    image = F.to_tensor(image)
    height = image.size(1)
    width = image.size(2)
    image_id = get_largest_id(anns["images"]) + 1
    file_name = os.path.basename(os.path.normpath(image_path))
    image_desc = {
        "coco_url": "",
        "date_captured": 0,
        "file_name": file_name,
        "flickr_url": "",
        "height": height,
        "id": image_id,
        "license": 0,
        "width": width
    }
    anns["images"].append(image_desc)
    
    ann_id = get_largest_id(anns["annotations"])
    for box in boxes:
        ann_id += 1
        box = convert_voc_to_coco(box)
        area = box[2]*box[3]

        new_ann = {
        "area": area,
        "attributes": {
            "occluded": False,
            "rotation": 0.0
        },
        "bbox": box,
        "category_id": 0,
        "id": ann_id,
        "image_id": image_id,
        "iscrowd": 0,
        "segmentation": []
        }         
        
        anns["annotations"].append(new_ann)
    
    return anns

@click.command()
@click.option('--image-path', type=click.Path(exists=True), required=True, help='Path to the image to be augmented')
@click.option('--annotation-path', type=click.Path(exists=True), required=True, help='Path to coco annotations')
@click.option('--num-of-crops', type=click.INT, default=0, help='Number of crops to be generated')
def main(image_path, annotation_path, num_of_crops):
    image = Image.open(image_path, mode= "r")
    image = image.convert("RGB")
    boxes, anns = parse_coco_annot(annotation_path, os.path.basename(os.path.normpath(image_path)))
    boxes = torch.FloatTensor(boxes)

    new_image, new_boxes = flip(image, boxes)
    new_path = image_path.split('.')[-2] + '_flip.jpg'
    new_image.save(new_path)
    
    new_anns = add_boxes_to_coco(new_boxes, anns, new_image, new_path)
    
    
    for i in range(num_of_crops):
        new_path = image_path.split('.')[-2] + '_crop' + str(i) + '.jpg'
        new_image,new_boxes= random_crop(image, boxes)
        draw_PIL_image(new_image, new_boxes, new_path)
        new_image.save(new_path)
        
        new_anns = add_boxes_to_coco(new_boxes, anns, new_image, new_path)
    
    with open(annotation_path, "w") as outfile:
        json.dump(new_anns, outfile, indent=2)
    


if __name__ == '__main__':
    main()

