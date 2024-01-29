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
from skimage.color import rgb2hed, hed2rgb

"""
This script reads in a folder of whole slide images or a single whole slide image and returns a tissue mask.
For H&E with pen_markerss, I would recommend "--based_on hed" and "--hed h"

example usage:
python TissueSegmentation.py --wsi_file_dir /path/to/wsi/files --based_on grey --grey_threshold 210 --red_threshold 200 --green_threshold 190 --blue_threshold 200 --contrast 5 --min_width 400 --resize True --save True --save_dir /path/to/save/directory --save_ending _mask.png --show False --show_detailed False
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Tissue Segmentation')
    parser.add_argument('--wsi_file_dir', type=str, default=None, help='path to the whole slide images')
    #parser.add_argument('--wsi_format', type=str, default='.svs', help='file format of the whole slide images, typically ".svs" or ".ndpi"')
    parser.add_argument('--wsi_file', type=str, default=None, help='path to a single whole slide image')
    parser.add_argument('--based_on', type=str, default='grey', help='grey works quite well for most stains; for h&e, you can also try "rgb" or "hed"')
    parser.add_argument('--hed', type=str, default='h', help='"h" for hematoxylin staining, "e" for eosin staining, "d" for dab staining (brown staining in general, so works for p53 and tff3)')
    parser.add_argument('--grey_threshold', type=int, default=210, help='threshold for the grey-based segmentation')
    parser.add_argument('--red_threshold', type=int, default=190, help='min red value; important, if "rgb" based and h&e staining, threshold for the rgb-based segmentation')
    parser.add_argument('--green_threshold', type=int, default=210, help='max green value; important, if "rgb" based and h&e staining, threshold for the rgb-based segmentation')
    parser.add_argument('--blue_threshold', type=int, default=190, help='min blue value; important, if "rgb" based and h&e staining, threshold for the rgb-based segmentation')
    parser.add_argument('--contrast', type=float, default=10, help='contrast increase; if the tissue is not well separated from the background, increase this value; you can easily go up to 20 or higher')
    parser.add_argument('--min_width', type=int, default=400, help='min width of the tissue mask; if lowest level has high resolution, then this value speeds up the process, as the size of the lowest level will be reduced to this value')
    parser.add_argument('--resize', type=bool, default=True, help='by default, the level of the whole slide image will be chosen that has the smallest with above the defined min width. However, sometimes already the low level can have very high resolution, which slows down the process massively. If True, then the mask will have a width of min_width, and a height that is scaled accordingly')
    parser.add_argument('--save', type=bool, default=True, help='whether to save the tissue mask')
    parser.add_argument('--save_dir', type=str, default=None, help='path to the directory where the tissue mask should be saved; if None, then it will be saved in the same directory as the whole slide image')
    parser.add_argument('--save_ending', type=str, default='_mask.png', help='file ending and format of the tissue mask, typically "_mask.png" or ".jpg"')
    parser.add_argument('--show', action='store_true', help='whether to show the plot of the tissue mask')
    parser.add_argument('--show_detailed', action='store_true', help='whether to show the plot of the tissue mask in different stages of algorithm')
    parser.add_argument('--min_pixel_count', type=int, default=30, help='minimum number of pixels for a polygon to be considered as tissue')
    parser.add_argument('--pen_markers', action='store_true', help='whether to expect pen_markers such as pen markers')
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

def filter_ihc(ihc, hed='e'):
    # Prepare the arguments for the filter
    # Rescale the image data to [0, 255]
    ihc = (ihc * 255).astype(np.uint8)
    # convert to HSV color space
    hsv = cv2.cvtColor(ihc, cv2.COLOR_RGB2HSV)
    # Define lower and upper bounds for the color
    if hed == 'e': # pink
        lower = np.array([150, 10, 20])
        upper = np.array([180, 255, 255])
    elif hed == 'd': # brown
        lower = np.array([8, 10, 20])
        upper = np.array([18, 255, 200])
    elif hed == 'h': # blue
        lower = np.array([90, 10, 20])
        upper = np.array([120, 255, 255])

    # Create a mask for the color
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND the mask and the original image
    res = cv2.bitwise_and(ihc, ihc, mask=mask)
    return cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

def get_mask(file_name, based_on='grey', contrast=3, grey_threshold=220,
             red_threshold=190, green_threshold=210, blue_threshold=190, min_width=200,
             resize=True, show=False, show_detailed=False, save=None, min_pixel_count=30,
             hed='h', pen_markers=False):
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
    wsi_level_grey = wsi_level.convert('L')
    if pen_markers:
        wsi_level_grey = np.array(wsi_level_grey).astype(np.float32)
        wsi_level_rgb = np.array(wsi_level_rgb).astype(np.float32)

        triangle_threshold = filters.threshold_triangle(wsi_level_grey.reshape(-1))
        otsu_threshold = filters.threshold_otsu(wsi_level_grey.reshape(-1))
        
        mask = wsi_level_grey < otsu_threshold
        selem = square(3)
        dilated_mask = dilation(mask, selem)
        #dilated_mask = mask

        #wsi_level_grey[wsi_level_grey > triangle_threshold] = 255
        #wsi_level_grey[wsi_level_grey < otsu_threshold] = 255

        # instead of white (255), make same color as background
        # we can set the background color to the first pixel of the image

        background_color = wsi_level_grey[0,0]
        wsi_level_grey[dilated_mask] = background_color

        #wsi_level_rgb[wsi_level_grey == background_color] = background_color
        #wsi_level_rgb = Image.fromarray(wsi_level_rgb.astype(np.uint8))
        wsi_level_grey = Image.fromarray(wsi_level_grey.astype(np.uint8))
    if based_on == 'rgb':
        enhancer = ImageEnhance.Contrast(wsi_level_rgb)
    elif based_on == 'grey':
        enhancer = ImageEnhance.Contrast(wsi_level_grey)
    elif based_on == 'hed':
        enhancer = ImageEnhance.Contrast(wsi_level_rgb) ##

    # increase contrast
    wsi_level_contrast = enhancer.enhance(contrast)

    # filter rgb
    if based_on == 'rgb':
        red, green, blue = wsi_level_rgb.split()
        red = np.array(red).astype(np.float32)
        green = np.array(green).astype(np.float32)
        blue = np.array(blue).astype(np.float32)
        rgb_mask = (red > red_threshold) & (green < green_threshold) & (blue > blue_threshold)

        red, green, blue = wsi_level_contrast.split()
        red = np.array(red).astype(np.float32)
        green = np.array(green).astype(np.float32)
        blue = np.array(blue).astype(np.float32)
        contrast_mask = (red > red_threshold) & (green < green_threshold) & (blue > blue_threshold)

    # filter grey
    grey_mask = np.array(wsi_level_contrast) > grey_threshold #np.array(wsi_level_grey) > 220
    # remove spots where grey_mask is 255 and replace with 0
    grey_mask = grey_mask.astype(np.uint8)
    grey_mask[grey_mask == 0] = 255
    if based_on == 'grey':
        contrast_mask = 1-grey_mask
    elif based_on == 'hed':
        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(wsi_level_rgb)

        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        if hed == 'h':
            contrast_mask = filter_ihc(ihc_h, hed=hed)
        elif hed == 'e':
            contrast_mask = filter_ihc(ihc_e, hed=hed)
        elif hed == 'd':
            contrast_mask = filter_ihc(ihc_d, hed=hed)
        # save image of contrast_mask
        cv2.imwrite('contrast_mask.png', contrast_mask)

    mask = contrast_mask.astype(np.uint8)
    inverted_mask = cv2.bitwise_not(mask)
    h,w = mask.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    filled_mask = cv2.bitwise_not(inverted_mask)
    height = filled_mask.shape[0]
    contours_filled_mask, _ = cv2.findContours(filled_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_filled_image = cv2.drawContours(np.zeros_like(filled_mask), contours_filled_mask, -1, 255, -1)
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

    if show_detailed:
        # plot wsi_level_rgb, wsi_level_hsv, wsi_level_grey with corresponding masks
        fig = plt.figure()
        plt.title(file_name)
        ax1 = fig.add_subplot(421)
        ax1.imshow(wsi_level_rgb)
        ax2 = fig.add_subplot(422)
        ax2.imshow(1-rgb_mask, cmap='gray')
        ax3 = fig.add_subplot(423)
        ax3.imshow(wsi_level_contrast)
        ax4 = fig.add_subplot(424)
        ax4.imshow(1-contrast_mask, cmap='gray')
        ax5 = fig.add_subplot(425)
        ax5.imshow(wsi_level_grey, cmap='gray')
        ax6 = fig.add_subplot(426)
        ax6.imshow(grey_mask, cmap='gray')
        ax7 = fig.add_subplot(427)
        ax7.imshow(contours_filled_image, cmap='Purples')
        ax8 = fig.add_subplot(428)
        # loop over polygons and plot them
        try:
            for polygon in merged_polygons: # polygons:
                x,y = polygon.exterior.xy
                y = height - np.array(y)
                random_color = (random.random(), random.random(), random.random())  # Generate a random RGB color
                ax8.fill(x, y, color=random_color)
        except:
            # empty image
            pass
            
        ax8.set_aspect('equal')
        plt.show()

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(6, 3))
        # Plot the original RGB image
        axs[0].imshow(wsi_level_rgb)
        axs[0].set_title('Original')
        # Plot the mask polygons
        axs[1].imshow(mask_polygons, cmap='gray')
        axs[1].set_title('Mask')
        # Plot the original RGB image with mask polygons overlaid
        axs[2].imshow(wsi_level_rgb)
        axs[2].imshow(mask_polygons, cmap='Purples', alpha=0.15)  # Overlaid with transparency
        axs[2].set_title('Overlay')
        plt.show()

    return merged_polygons

if __name__ == '__main__':
    args = parse_args()
    wsi_file_dir = args.wsi_file_dir
    wsi_file = args.wsi_file
    based_on = args.based_on
    hed = args.hed
    grey_threshold = args.grey_threshold
    red_threshold = args.red_threshold
    green_threshold = args.green_threshold
    blue_threshold = args.blue_threshold
    contrast = args.contrast
    min_width = args.min_width
    resize = args.resize
    save = args.save
    save_dir = args.save_dir
    save_ending = args.save_ending
    show = args.show
    show_detailed = args.show_detailed
    min_pixel_count = args.min_pixel_count
    pen_markers = args.pen_markers

    print('Tissue Segmentation')

    if save_dir is None:
        save_dir = wsi_file_dir
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Created directory: ', save_dir)

    if wsi_file_dir is not None:
        print('Reading files from directory: ', wsi_file_dir)
        # Read all files in the directory
        files = [os.path.join(wsi_file_dir, file) for file in os.listdir(wsi_file_dir) if file.endswith('.svs') or file.endswith('.ndpi')]
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
            save_path = os.path.join(save_dir, ''.join(os.path.basename(file).split('.')[:-1])+save_ending)
        else:
            save_path = None
        get_mask(file, based_on=based_on, contrast=contrast, grey_threshold=grey_threshold,
                 red_threshold=red_threshold, green_threshold=green_threshold, blue_threshold=blue_threshold,
                 min_width=min_width, resize=resize, show=show, show_detailed=show_detailed, save=save_path,
                 min_pixel_count=min_pixel_count, hed=hed, pen_markers=pen_markers)
    end_time = time.time()
    # Calculate elapsed time in seconds
    elapsed_time_sec = end_time - start_time

    # Convert elapsed time to minutes and seconds
    elapsed_time_min = int(elapsed_time_sec // 60)
    elapsed_time_sec = int(elapsed_time_sec % 60)

    print('Finished processing ', len(files), ' files in ', elapsed_time_min, ' minutes and ', elapsed_time_sec, ' seconds')
