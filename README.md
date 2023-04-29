# Image Analysis Scripts

## Libraries Used 

- os library: The os library provides a way of interacting with the operating system. It is used in the script to read file paths, check file existence, create directories, and execute shell commands.

- csv library: The csv library provides functionality to read from and write to CSV files. It is used in the IOUMatrix.py script to read CSV files containing localizer-generated bounding boxes and ground truth bounding boxes.

- numpy library: The numpy library provides numerical computing capabilities in Python. It is used extensively in the script for manipulating arrays and performing mathematical calculations.

- PIL library: The PIL library (Python Imaging Library) is a library that adds support for opening, manipulating, and saving many different image file formats. It is used in the script to read and manipulate images.

- matplotlib library: The matplotlib library is a plotting library for the Python programming language. It is used in the BoundingBoxAnalysis.py script to plot histograms of the false positive detections.

- pandas library: The pandas library is a library for data manipulation and analysis. It is used in the IOUMatrix.py script to create DataFrames to store the bounding boxes and IOU matrixes.

- torch library: The torch library is a library for building and training neural networks. It is used in the IOUMatrix.py script to calculate the Intersection over Union (IoU) between bounding boxes.

- torchvision library: The torchvision library is a library that provides access to popular datasets, model architectures, and image transformations for computer vision. It is used in the IOUMatrix.py script to calculate the Intersection over Union (IoU) between bounding boxes.

- collections library: The collections library provides alternative implementations of Python's built-in container types. It is used in the BoundingBoxAnalysis.py script to create dictionaries to store the false and true positive detections.

- argparse library: The argparse library provides a way to parse command-line arguments in Python. It is used in the main.py script to parse arguments for the script.

## main.py -> Main Script Execution

Calls all of the image analysis scripts in the following order:

1. BoundingBoxAnalysis.main(): This is used to generate dictionaries of all false positive detections and all true positive detections:

   - This script implicity makes use of IOUMatrix.py to calculate the IOUMatrixes for each image in the sample set


## Detailed Descriptions for each Python File:

### IOUMatrix.py

Please keep in mind the following:
  1. All dictionary objects in this script follow the following format: <img_name: [array of bounding boxes]>.
  2. This script utlizes 2 dictionaries: gt_dict and l_dict
  3. gt_dict is a dictionary of ground truth bounding boxes for each image
  4. l_dict is a dictionary of ground truth bounding boxes for each image

The script has three main functions that are used to calculate Intersection over Union (IoU) between ground truth bounding boxes (gt) and localizer generated bounding boxes (l) for each image in a sample set.

1. clean_localization_dict(l_dict): This function removes non-animal class rows and the header ['x', 'y', 'w', 'h'] from the box list for each file in the given l_dict dictionary. The cleaned dictionary new_dict contains bounding boxes for each image with an animal in the sample set.

2. generate_localization_dict(): This function generates a dictionary of localizer generated bounding boxes (l) for each image in the sample set. It iterates through a folder of CSV files and creates a dictionary with the filename as the key and a list of boxes as the value. The boxes list contains coordinates in the format [x, y, w, h].

3. iOU(localizer, ground_truth): This function calculates the IoU between each pair of bounding boxes in the localizer and ground_truth arrays using the torchvision.ops.box_iou function. It returns the IoU matrix for a single image as a Pandas DataFrame.

Additionally, there are two more functions:

4. generatetruePositivityArray(bestBoundingBox): This function calculates and returns the ratio of good bounding boxes with an IoU greater than 0.5 to the total number of localizer generated bounding boxes for each image in the sample set. The bestBoundingBox input parameter is the list of all the IoU matrixes from the sample set.

5. main(): This function is not described explicitly, but it likely runs the script and returns a list of all the IoU matrixes from the sample set. The function returns the following tuple (iou_mat_list, gt_dict, l_dict).

### BoundingBoxAnalysis.py

The script generates a dictionary of false and true positive detections in a sample set of images. The script calls the IOUMatrix.py main() function to calculate the IOU matrixes for each image in the sample. Then, the script generates a dictionary of false positive detections and true positive detections in the sample. The script plots a histogram with the resolution of localizer bounding boxes on the x-axis and frequency of false positive detections on the y-axis. The script draws ground truth and localizer bounding boxes on all the images. If someone plans on changing the image size, they need to change the normalization in the script GroundTruthBoxes.py. The script has three main functions:

1. falseDetectionsAnalysis(): This function produces a false positive detections dictionary, true positive detections dictionary, and the number of localizer detections across all images. The function loops through the IOU matrices and creates dictionaries for false and true detections. It initializes a variable for the total number of detections and loops through the columns and rows in the IOU matrix. If there are no matches for the current column, the function adds it to the false detection dictionary. Otherwise, it adds it to the true detection dictionary.

2. plot_Detections(): This function plots a histogram of the number of false detections against the resolution of localizer bounding boxes. The function creates arrays of tuples with coordinates for false and true positive detections. It loops through the dictionaries of false and true detections, calculates the number of pixels in a bounding box, and appends it to the arrays. Then, the function creates a histogram with 20 bins, sets the title and axis labels, and displays the histogram.

3. drawBoundingBoxes(): This function draws ground truth and localizer bounding boxes on all the images. It loops through the IOU matrices and draws the ground truth bounding boxes in blue and localizer bounding boxes in red on the images.

4. main():
   Here's what the script does as a whole:
  
   - The script calls the IOUMatrix.py module's main() function to calculate the IOU matrices for each image in the sample.
   - It generates a dictionary of false positive detections and true positive detections in the sample set of images using the IOU matrices.
   - It plots a histogram of the frequency of false positive detections by bounding box resolution.
   - It draws the ground truth and localizer bounding boxes on all the images.
    - The main script calls the falseDetectionsAnalysis(), plot_Detections(), and drawBoundingBoxes() functions to perform these tasks.





