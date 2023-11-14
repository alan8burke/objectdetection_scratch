from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_filenames_fromdir(
    filepath: Union[str, os.PathLike], folder: str
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Parameters
    ----------
    filepath : Union[str, os.PathLike]
        The path to the directory containing the images and labels.
    folder : str
        The name of the folder containing the images and labels.

    Returns
    -------
    Tuple[Dict[int, str], Dict[int, str]]
        A tuple containing two dictionaries:
        - The first dictionary maps image indices to filenames.
        - The second dictionary maps label indices to filenames.
    """
    files = os.listdir(filepath / folder)
    filename_images = dict()
    filename_labels = dict()
    index = 1

    for f in files:
        if f.endswith(".jpg"):
            filename_images[index] = f
        else:
            filename_labels[index] = f
            index += 1

    print(folder, " : ", index - 1, "images")
    return filename_images, filename_labels


def extract_boxes(
    filename_xml: Union[str, os.PathLike]
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    """
    Extract bounding boxes and image dimensions from an XML file.

    Parameters
    ----------
    filename_xml : str or os.PathLike
        The path to the XML file.

    Returns
    -------
    boxes_mask : List[List[int]]
        A list of bounding boxes for objects with the "mask" label.
    boxes_nomask : List[List[int]]
        A list of bounding boxes for objects with the "nomask" label.
    width : int
        The width of the image.
    height : int
        The height of the image.
    """
    # load and parse the file
    tree = ET.parse(filename_xml)

    # get the root of the document
    root = tree.getroot()

    # extract each bounding box
    boxes_nomask = list()
    boxes_mask = list()
    for box in root.findall(".//object"):
        xmin = int(box.find("bndbox/xmin").text)
        ymin = int(box.find("bndbox/ymin").text)
        xmax = int(box.find("bndbox/xmax").text)
        ymax = int(box.find("bndbox/ymax").text)
        coors = [xmin, ymin, xmax, ymax]

        if str(box.find("name").text) == "mask":
            boxes_mask.append(coors)
        else:
            boxes_nomask.append(coors)

    # extract image dimensions
    width = int(root.find(".//size/width").text)
    height = int(root.find(".//size/height").text)
    return boxes_mask, boxes_nomask, width, height


def display_images_withlabels(filename_img: Union[str, os.PathLike]) -> None:
    """
    Display an image with its corresponding labels for Jupyter Notebooks

    Parameters
    ----------
    filename_img : Union[str, os.PathLike]
        The path to the image file.
    """
    img = cv2.imread(str(filename_img))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert str ending from jpg to xml
    boxes_mask, boxes_nomask, __, __ = extract_boxes(filename_img.with_suffix(".xml"))

    for Xmin, Ymin, Xmax, Ymax in boxes_mask:
        cv2.rectangle(rgb_img, (Xmin, Ymin), (Xmax, Ymax), (0, 255, 0), 4)  # Green

    for Xmin, Ymin, Xmax, Ymax in boxes_nomask:
        cv2.rectangle(rgb_img, (Xmin, Ymin), (Xmax, Ymax), (255, 0, 0), 4)  # Red

    plt.imshow(rgb_img)
    plt.show()


def display_multipleplots(
    images_list: List[np.ndarray],
    title_list: List[str],
    columns: int,
    rows: int,
    figsize: Tuple[int, int],
) -> None:
    """
    Display multiple plots in a grid layout.

    Parameters
    ----------
    images_list : List[np.ndarray]
        A list of images to plot.
    title_list : List[str]
        A list of titles for each plot.
    columns : int
        The number of columns in the grid layout.
    rows : int
        The number of rows in the grid layout.
    figsize : Tuple[int, int]
        The size of the figure.
    """
    # Plotting
    fig = plt.figure(figsize=figsize)

    for index_i, image_to_plot in enumerate(images_list):
        fig.add_subplot(rows, columns, index_i + 1)
        plt.imshow(image_to_plot)
        plt.title(title_list[index_i])
        plt.axis("on")


def display_RGBimageANDMask(
    img_path: Union[str, os.PathLike], mask_number: int, threshold: int
) -> None:
    """
    Display an RGB image and its corresponding mask, with post-processing steps applied.

    Parameters
    ----------
    img_path : Union[str, os.PathLike]
        The path to the RGB image file.
    mask_number : int
        The number of the mask to be displayed.
    threshold : int
        The threshold value for the mask.
    """
    MASK_NUMBER = mask_number
    THRESH = threshold

    # Extract boxes coordinates
    boxes_mask, __, __, __ = extract_boxes(img_path.with_suffix(".xml"))
    coords = boxes_mask[MASK_NUMBER - 1]
    Xmin, Ymin, Xmax, Ymax = coords[0], coords[1], coords[2], coords[3]

    # read image and convert to various formats
    img = cv2.imread(str(img_path))  # BGR format (full image)
    bgr_img = img[Ymin:Ymax, Xmin:Xmax]
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # converts to RGB format
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)  # converts to HSV

    # Apply blur to remove noise
    blurred_hsv_img = cv2.blur(hsv_img, (3, 3))

    # HSV average values of a face mask
    H_avg, S_avg, V_avg = 21, 12, 178

    # Threshold the HSV image to get only average HSV color from the mask region
    thresh = THRESH  # 40
    minHSV = np.array([H_avg - thresh, S_avg - thresh, V_avg - thresh])
    maxHSV = np.array([H_avg + thresh, S_avg + thresh, V_avg + thresh])
    maskHSV = cv2.inRange(blurred_hsv_img, minHSV, maxHSV)
    # Bitwise-AND mask and original image
    # resultHSV = cv2.bitwise_and(blurred_hsv_img, blurred_hsv_img, mask=maskHSV)

    # Extract biggest object of the lot
    cnts, _ = cv2.findContours(maskHSV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(maskHSV.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(maskHSV, out)

    # Removing objects inside the mask
    output_mask = cv2.morphologyEx(
        out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    )
    rgb_mask = cv2.bitwise_and(rgb_img, rgb_img, mask=output_mask)

    # Plotting
    images_to_plot = [rgb_img, maskHSV, rgb_mask, output_mask]
    images_title = [
        "Input RGB image",
        "Mask after color segmentation",
        "Mask RGB after post processing",
        "Mask after post processing steps",
    ]
    display_multipleplots(
        images_to_plot, images_title, columns=2, rows=2, figsize=(15, 10)
    )
