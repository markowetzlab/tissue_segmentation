# Tissue Segmentation

This script takes whole slide images (typically .svs or .ndpi files; either from biospsies or Cytosponge etc) as an input and returns an image file (typically .png file) of the mask of where the tissue is located (black: tissue, white: background). The script runs really fast and does not need any gpu capacity as it relies on basic pixel filtering.

The script works both with specific files (`wsi_file`) or file directories (`wsi_file_dir`), where it filters for files ending with either '.svs' or '.ndpi'.
There are two different ways of filtering: Either filtering on the grey scale (not ideal with lots of artifacts, as they will also be treated as tissue; works well with all stainings), or rgb scale (developed for H&E slides with lots of artifacts). For the grey scale, one can manually change the threshold of the pixels that will be filtered (`grey_threshold`). For the rgb scale, the default thresholds work well for H&E slides (`red_threshold`,`green_threshold`,`blue_threshold`). These threshold are per default for the Cytosponge. For biospsies, they can vary heavily.
One important parameter is `contrast`. Depending on how the staining was performed and with dye was used, good values for contrasts can vary from 3 to 30. As H&E stains are usually more intense, 3 is a good number to start with. For grey stains such as TP53, 20 is a good number to start with.
For finding the right thresholds and contrast, one can look at some examples using the `show` parameter.
The output of this parameter is shown here, where the mask is the output of the script:
- Example of H&E staining using rgb filtering with default thresholds and contrast=3![](rgb_example.png)
- Example of TFF3 staining using grey filtering with default threshold and contrast=20
![](grey_example.png)

.svs and .ndpi files can have a verying number of levels with very different resolutions even in the last level (aka the most zoomed out ones). Automatically, the highest level where the width is above `min_width` is chosen. However, if the level has a very high resolution, then the script will take longer to run. Usually, a width of 400 pixels is enough to get good outlines for the tissue. When specifying `resize`, then instead of the level's resolution, the size of the mask will be resized to the width of `min_width` and the height will automatically be adjusted to fit the original proportions.


## Requirements

Python 3.6 or later with the following packages:

- `argparse`
- `numpy`
- `matplotlib`
- `Pillow`
- `Shapely`
- `opencv-python`
- `tqdm`
- `torch`
- `openslide-python`

The requirements can be installed with pip:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script using the following command:
```bash
python TissueSegmentation.py --wsi_file_dir <path_to_images> [options]
```

### Options
`--wsi_file_dir`: Path to the directory containing the whole slide images.
`--wsi_file`: Path to a single whole slide image.
`--based_on`: Basis for the segmentation. Can be 'grey' or 'rgb'. Default is 'grey'.
`--grey_threshold`: Threshold for the grey-based segmentation. Default is 210.
`--red_threshold`: Minimum red value for the RGB-based segmentation. Default is 190.
`--green_threshold`: Maximum green value for the RGB-based segmentation. Default is 210.
`--blue_threshold`: Minimum blue value for the RGB-based segmentation. Default is 190.
`--contrast`: Contrast increase. Default is 10.
`--min_width`: Minimum width of the tissue mask. Default is 400.
`--resize`: Whether to resize the mask to have a width of min_width. Default is True.
`--save`: Whether to save the tissue mask. Default is True.
`--save_dir`: Directory where the tissue mask should be saved. If None, the mask will be saved in the same directory as the whole slide image.
`--save_ending`: File ending and format of the tissue mask. Default is '_mask.png'.
`--show`: Whether to show the plot of the tissue mask.
`--show_detailed`: Whether to show the plot of the tissue mask in different stages of the algorithm.

### Example

```bash
python TissueSegmentation.py --wsi_file_dir /path/to/images --contrast 10 --resize --save --show
```