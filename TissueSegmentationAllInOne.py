import argparse
import os
import openslide
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from skimage import filters
from skimage.morphology import dilation, disk, square
import cv2
import time
from tqdm import tqdm
import random
from skimage.color import rgb2hed, hed2rgb, rgb2hsv, hsv2rgb, rgb2gray

"""
This script reads in a folder of whole slide images or a single whole slide image and returns a tissue mask.
For H&E with pen_markerss, I would recommend "--based_on hed" and "--hed h"

example usage:
python TissueSegmentation.py --wsi_file_dir /path/to/wsi/files --based_on gray --gray_threshold 210 --red_threshold 200 --green_threshold 190 --blue_threshold 200 --contrast 5 --min_width 400 --resize True --save True --save_dir /path/to/save/directory --save_ending _mask.png --show False --show_detailed False
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Tissue Segmentation')
    parser.add_argument('--wsi_dir', type=str, default=None, help='path to the whole slide images')
    #parser.add_argument('--wsi_format', type=str, default='.svs', help='file format of the whole slide images, typically ".svs" or ".ndpi"')
    parser.add_argument('--wsi_file', type=str, default=None, help='path to a single whole slide image')
    parser.add_argument('--gray_threshold', type=int, default=210, help='threshold for the gray-based segmentation')
    parser.add_argument('--hed_contrast', type=float, default=1, help='contrast increase; if the tissue is not well separated from the background, increase this value; you can easily go up to 20 or higher')
    parser.add_argument('--gray_contrast', type=float, default=20, help='contrast increase; if the tissue is not well separated from the background, increase this value; you can easily go up to 20 or higher')
    parser.add_argument('--min_width', type=int, default=1200, help='min width of the tissue mask; if lowest level has high resolution, then this value speeds up the process, as the size of the lowest level will be reduced to this value')
    parser.add_argument('--resize', type=bool, default=True, help='by default, the level of the whole slide image will be chosen that has the smallest with above the defined min width. However, sometimes already the low level can have very high resolution, which slows down the process massively. If True, then the mask will have a width of min_width, and a height that is scaled accordingly')
    parser.add_argument('--save', type=bool, default=True, help='whether to save the tissue mask')
    parser.add_argument('--save_dir', type=str, default=None, help='path to the directory where the tissue mask should be saved; if None, then it will be saved in the same directory as the whole slide image')
    parser.add_argument('--save_ending', type=str, default='_tissuetector.png', help='file ending and format of the tissue mask, typically "_mask.png" or ".jpg"')
    #parser.add_argument('--show', action='store_true', help='whether to show the plot of the tissue mask')
    #parser.add_argument('--show_detailed', action='store_true', help='whether to show the plot of the tissue mask in different stages of algorithm')
    parser.add_argument('--min_pixel_count', type=int, default=25, help='minimum number of pixels for a polygon to be considered as tissue')
    parser.add_argument('--pen_markers', action='store_true', help='whether to expect pen_markers such as pen markers')
    parser.add_argument('--he_only', action='store_true')
    parser.add_argument('--gray_only', action='store_true')
    parser.add_argument('--blue_marker', action='store_true')
    parser.add_argument('--green_marker', action='store_true')
    parser.add_argument('--red_marker', action='store_true')
    parser.add_argument('--black_marker', action='store_true')
    parser.add_argument('--filter_kernel', default=5, type=int, help='filter kernel for black and red pen markers, default=5 leads to kernel of (5,5)')
    parser.add_argument('--dilute', default=True, help='increasing the outline of the pen markers')
    parser.add_argument('--he_cutoff_percent', default=5, type=int, help='if there is less than the cutoff percent of he tissue, instead of hed space, the gray color space will be used')
    args = parser.parse_args()
    return args

def polygons_to_mask(polygons, image_shape):
    # Create a blank image with the same shape as the original image
    mask = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)

    # Loop over each polygon
    for polygon in polygons:
        # Get the x and y coordinates of the polygon
        x, y = polygon.exterior.coords.xy

        # Convert the polygon coordinates to integer
        poly_coords = np.array([list(zip(x, y))], dtype=np.int32)

        # Fill the polygon area in the mask with 1
        cv2.fillPoly(mask, poly_coords, 1)

    return mask

def get_hetissue(wsi_rgb, hed_contrast):
    enhancer = ImageEnhance.Contrast(wsi_rgb)
    # increase contrast
    wsi_rgb = enhancer.enhance(hed_contrast)
    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(wsi_rgb)
    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc = hed2rgb(np.stack((ihc_hed[:, :, 1], null, null), axis=-1)) # we are only intrested in 'e' of 'hed'
    contrast_mask = ihc.copy()
    contrast_mask[contrast_mask>0.99] = 0
    contrast_mask[contrast_mask!=0] = 1
    contrast_mask = rgb2gray(contrast_mask)
    # Count the number of 1s in the mask
    count_ones = np.count_nonzero(contrast_mask)
    # Get the total number of elements in the mask
    total_elements = contrast_mask.size
    # Calculate the ratio of 1s
    coverage = np.round(count_ones / total_elements * 100)
    #print(count_ones, total_elements, coverage)
    return contrast_mask, coverage

def get_graytissue(wsi_gray, gray_contrast, gray_threshold):
    enhancer = ImageEnhance.Contrast(wsi_gray)
    # increase contrast
    wsi_contrast = enhancer.enhance(gray_contrast)
    # filter gray
    gray_mask = np.array(wsi_contrast) > gray_threshold #np.array(wsi_level_gray) > 220
    # remove spots where gray_mask is 255 and replace with 0
    gray_mask = gray_mask.astype(np.uint8)
    gray_mask[gray_mask == 0] = 255
    contrast_mask = 1-gray_mask
    return contrast_mask


def get_mask(file_name, hed_contrast=2, gray_contrast=10, gray_threshold=220, min_width=1200,
             resize=True, save=None, min_pixel_count=30, blue_marker=False,
             red_marker=False, black_marker=False, green_marker=False, hed=True, gray=True, kernel=(5,5), dilute=True, he_cutoff_percent=5):
    wsi = openslide.OpenSlide(file_name)
    # loop over levels to find level with width > min_width
    level = wsi.level_count - 1
    while wsi.level_dimensions[level][0] < min_width:
        print('level dimensions: ', wsi.level_dimensions[level], 'for level: ', level)
        level -= 1
    print('level picked: ', level, 'with dimensions: ', wsi.level_dimensions[level])

    wsi_level = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    if resize:
        aspect_ratio = wsi_level.size[1] / wsi_level.size[0]
        new_height = int(min_width * aspect_ratio)
        wsi_level = wsi_level.resize((min_width, new_height))
        print('resized to: ', wsi_level.size)
    wsi_level_rgb = wsi_level.convert('RGB')
    wsi_level_gray = wsi_level.convert('L')
    wsi_level_hsv = rgb2hsv(np.array(wsi_level)[:,:,:3])

    # get he or gray tissue
    if hed is True:
        tissue_mask, coverage = get_hetissue(wsi_level_rgb, hed_contrast)
        if gray is True:
            if coverage < he_cutoff_percent or coverage == 100.0:
                tissue_mask = get_graytissue(wsi_level_gray, gray_contrast, gray_threshold)
    elif gray is True:
        tissue_mask = get_graytissue(wsi_level_gray, gray_contrast, gray_threshold)

    #plt.imshow(tissue_mask, cmap='gray')
    #plt.title(coverage)
    #plt.show()

    wsi_level_hsv = (wsi_level_hsv * 255).astype('uint8')
    pen_mask = np.zeros_like(tissue_mask)
    if black_marker:
        black = cv2.inRange(wsi_level_hsv, np.array([0, 0, 0]).astype('uint8'), np.array([255, 255, 165]).astype('uint8'))
        # filter black mask
        black_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, black_kernel)
        _, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pen_mask = cv2.bitwise_or(pen_mask.astype('uint8'), black.astype('uint8'))
    if blue_marker:
        blue = cv2.inRange(wsi_level_hsv, np.array([130, 50, 30]).astype('uint8'), np.array([180,255,255]).astype('uint8'))
        pen_mask = cv2.bitwise_or(pen_mask.astype('uint8'), blue.astype('uint8'))
    if green_marker:
        green = cv2.inRange(wsi_level_hsv, np.array([30, 30, 50]).astype('uint8'), np.array([130, 255, 255]).astype('uint8'))
        pen_mask = cv2.bitwise_or(pen_mask.astype('uint8'), green.astype('uint8'))
    if red_marker:
        red1 = cv2.inRange(wsi_level_hsv, np.array([0, 30, 30]).astype('uint8'), np.array([30, 255, 255]).astype('uint8'))
        red2 = cv2.inRange(wsi_level_hsv, np.array([200, 100, 100]).astype('uint8'), np.array([255, 255, 255]).astype('uint8'))
        red = cv2.bitwise_or(red1,red2)
        # filter red mask
        red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, red_kernel)
        _, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pen_mask = cv2.bitwise_or(pen_mask.astype('uint8'), red.astype('uint8'))

    # dilute pen markers
    if dilute:
        selem = square(int(min_width/400)) #TODO
        pen_mask = dilation(pen_mask, selem)

    #print('tissue mask shape',tissue_mask.shape)
    #print(np.unique(tissue_mask))
    #print('pen mask shape',pen_mask.shape)
    #print(np.unique(pen_mask))
    tissue_mask_normalized = cv2.normalize(tissue_mask, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    tissue_mask = np.bitwise_and(tissue_mask_normalized.astype('int'), np.bitwise_not(pen_mask.astype('int')))
    #plt.imshow(tissue_mask)
    #plt.show()
    filled_mask = tissue_mask.astype(np.uint8)
    #inverted_mask = cv2.bitwise_not(mask)
    #h,w = mask.shape[:2]
    #mask = np.zeros((h+2,w+2),np.uint8)
    #filled_mask = cv2.bitwise_not(inverted_mask)
    #height = filled_mask.shape[0]
    contours_filled_mask, _ = cv2.findContours(filled_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours_filled_image = cv2.drawContours(np.zeros_like(filled_mask), contours_filled_mask, -1, 255, -1)
    # plot contours_filled_image
    #plt.imshow(contours_filled_image, cmap='Purples')
    polygons = []
    for contour in contours_filled_mask:
        if len(contour) >= min_pixel_count:
            polygon = Polygon(contour.reshape(-1, 2))
            polygons.append(polygon)
    buffered_polygons = [polygon.buffer(3) for polygon in polygons]
    merged_polygons = unary_union(buffered_polygons)
    if merged_polygons.geom_type == 'Polygon':
        merged_polygons = [merged_polygons]
    if isinstance(merged_polygons, MultiPolygon):
        merged_polygons = list(merged_polygons.geoms)
    if isinstance(merged_polygons, GeometryCollection):
        merged_polygons = list(merged_polygons.geoms)
    print('Number of polygons: ', len(merged_polygons))
    
    mask_polygons = polygons_to_mask(merged_polygons, wsi_level.size)
    if save is not None:
        # Scale mask values to 8-bit range
        mask_polygons_8bit = (mask_polygons * 255).astype(np.uint8)
        cv2.imwrite(save, mask_polygons_8bit)
        print('Saved mask to: ', save)

    return merged_polygons

if __name__ == '__main__':
    args = parse_args()
    wsi_file_dir = args.wsi_dir
    wsi_file = args.wsi_file
    #hed = args.hed
    gray_threshold = args.gray_threshold
    hed_contrast = args.hed_contrast
    gray_contrast = args.gray_contrast
    min_width = args.min_width
    resize = args.resize
    save = args.save
    save_dir = args.save_dir
    save_ending = args.save_ending
    #show = args.show
    #show_detailed = args.show_detailed
    min_pixel_count = args.min_pixel_count
    pen_markers = args.pen_markers
    blue_marker = args.blue_marker
    red_marker = args.red_marker
    black_marker = args.black_marker
    green_marker = args.green_marker
    he_only = args.he_only
    gray_only = args.gray_only
    filter_kernel = args.filter_kernel
    dilute = args.dilute
    he_cutoff_percent = args.he_cutoff_percent

    print('Tissue Segmentation')

    if save_dir is None:
        save_dir = wsi_file_dir
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Created directory: ', save_dir)

    if he_only is False and gray_only is False:
        he, gray = True, True
    elif he_only is True and gray_only is False:
        he, gray = True, False
    elif he_only is False and gray_only is True:
        he, gray = False, True
    else:
        he, gray = True, True

    if pen_markers:
        blue_marker, black_marker, red_marker, green_marker = True, True, True, True

    if wsi_file_dir is not None:
        print('Reading files from directory: ', wsi_file_dir)
        # Read all files in the directory
        files = [os.path.join(wsi_file_dir, file) for file in os.listdir(wsi_file_dir) if file.endswith('.svs') or file.endswith('.ndpi') or file.endswith('.tiff') or file.endswith('.tif')]
        print('Number of files: ', len(files))
    elif wsi_file is not None:
        print('Reading file: ', wsi_file)
        files = [wsi_file]
    else:
        print('Please specify either a directory or a single file')
        exit()

    start_time = time.time()
    for file in tqdm(files):
        print('Processing file: ', file)
        if save:
            save_path = os.path.join(save_dir, '.'.join(os.path.basename(file).split('.')[:-1])+save_ending)
        else:
            save_path = None
        get_mask(file, hed_contrast=hed_contrast, gray_contrast=gray_contrast, gray_threshold=gray_threshold, min_width=min_width,
             resize=resize, save=save_path, min_pixel_count=min_pixel_count, blue_marker=blue_marker,
             red_marker=red_marker, black_marker=black_marker, green_marker=green_marker, hed=he, gray=gray, kernel=(filter_kernel,filter_kernel), dilute=dilute, he_cutoff_percent=he_cutoff_percent)
    end_time = time.time()
    # Calculate elapsed time in seconds
    elapsed_time_sec = end_time - start_time

    # Convert elapsed time to minutes and seconds
    elapsed_time_min = int(elapsed_time_sec // 60)
    elapsed_time_sec = int(elapsed_time_sec % 60)

    print('Finished processing ', len(files), ' files in ', elapsed_time_min, ' minutes and ', elapsed_time_sec, ' seconds')
