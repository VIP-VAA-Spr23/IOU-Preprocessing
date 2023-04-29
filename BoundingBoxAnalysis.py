# Import Necessary Libraries
import IOUMatrix
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import json
import cv2


'''Script Description:
   This script is used to generate the dictionary of false positives detections and true positive 
   detections in the sample set of images. 

   The Script's main function works in the following fashion:

   1) Calls the IOUMatrix.py main() function to calculate the IOU matrixes for each image in the sample
   2) Generates the false positive detections and true positive dictionary for the sample 
   3) Graphing a histogram with resolution of localizer bounding boxes on x-axis and frequency of false positive detections on y-axis
   4) Draws ground truth and localizer bounding boxes on all the images

   Please refer to the main script for more instructions

   Please note that each image has been scaled to 1280 x 720 pixels and has previously been normalized 
   using these values.

   If you plan on changing the image size, please be sure to ammend the following functions according:
   line 148, 159 in drawingBoundingBoxes() and line 98, 107 in plotDetections().

   You will also have to accordingly change the normalization in the script GroundTruthBoxes.py. Please
   read comments in the script for more instructions
'''

def falseDetectionsAnalysis(IOU_mat_list, gt_dict, l_dict):

    '''Method produces 3 things from the set of images: false_positivies dictionary, true_positive dictionary 
       and number of localizer detections across all images'''

    # Get the list of images
    image_list = list(gt_dict.keys())

    # Create dictionaries for false and true detections
    false_detection_dict = {} # Dictionary of false positive detections
    true_detection_dict = {} # dictionary of true positive detections

    # Initialize a variable for the total number of detections
    total_detections = 0

    # Loop through the IOU matrices
    for i in range(len(IOU_mat_list)):

        # Add the number of columns in the IOU matrix to the total number of detections
        total_detections += len(IOU_mat_list[i].columns)

        # Loop through the columns in the IOU matrix
        for j in range(len(IOU_mat_list[i].columns)):
            
            # Initialize a variable for the number of matches
            no_match = 0

            # Loop through the rows in the IOU matrix
            for k in range(len(IOU_mat_list[i])):
        
                # Get the IOU value for the current item
                item = IOU_mat_list[i].iloc[k,j]

                # If the IOU value is zero or less than 0.5, increment the no_match variable
                if item == 0 or item < 0.5:
                    no_match += 1 
                
            # If there were no matches for the current column, add it to the false detection dictionary
            if no_match == len(IOU_mat_list[i].index):

                if image_list[i] not in false_detection_dict:

                    false_detection_dict[image_list[i]] = [l_dict[image_list[i]][j]]
                
                else:
                    false_detection_dict[image_list[i]].append(l_dict[image_list[i]][j]) 

            # Otherwise, add it to the true detection dictionary
            else:

                if image_list[i] not in true_detection_dict:

                    true_detection_dict[image_list[i]] = [l_dict[image_list[i]][j]]
                
                else:
                    true_detection_dict[image_list[i]].append(l_dict[image_list[i]][j]) 

    # Return the false detection dictionary, true detection dictionary, and total number of detections
    return false_detection_dict, true_detection_dict

def plot_Detections(false_detection_dict, true_detection_dict):

    '''This method plots a histogram of the number of false detections against the 
    resolution of localizer bounding boxes'''

    false_detections_arr = [] # array of tuples with coordinates for false positive detections
    true_detections_arr = []  # array of tuples with coordinates for true positive detections
    
    for key in false_detection_dict:

        for tup in false_detection_dict[key]:
            
            # calculating the number of pixels in a bounding boxes
            box_pixels = (tup[2]*1280 - tup[0]*1280) * (tup[3]*720 - tup[1]*720) 

            false_detections_arr.append(box_pixels)
    
    for key in true_detection_dict:

        for tup in true_detection_dict[key]:

            # calculating the number of pixels in a bounding boxes
            box_pixels = (tup[2]*1280 - tup[0]*1280) * (tup[3]*720 - tup[1]*720) 
            # pdb.set_trace()
            true_detections_arr.append(box_pixels)

    # Create a histogram with 20 bins
    plt.hist(false_detections_arr, bins=20, alpha=0.5)

    # Set the title and axis labels

    plt.title('Frequency of False Detections by Bounding Box Resolution')
    plt.xlabel('Resolution')
    plt.ylabel('Frequency')

    # Display the histogram
    plt.savefig('Frequency_of_False_Detections.png')
    plt.close()

    return len(false_detections_arr)

def print_stats(total_detections, false_detections):

    print("Total Detections: %d" %(total_detections))
    print("False Detections: %d" %(false_detections))

def drawBoundingBoxes(IOU_mat_list, false_detections, gt_dict, l_dict, image_list):

    '''This is a method to draw bounding boxes on all images in folder labeled Marked Images'''

    for i in range (len(IOU_mat_list)):

        key = image_list[i]

        arr_gt = gt_dict[key]
        arr_l = l_dict[key]
    
        image = cv2.imread('Check_Files/' + key + '.jpg')

        # gt bounding box drawing
        for k in range(len(arr_gt)):

            # Define the bounding box coordinates
            x1, y1, x2, y2 = int(arr_gt[k][0]*1280), int(arr_gt[k][1]*720), int(arr_gt[k][2]*1280), int(arr_gt[k][3]*720)
            #x1, y1, x2, y2 = 100, 100, 200, 200

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2),(0, 255, 0), 2)
    
        # localizer bounding box

        for i in range(len(arr_l)):

            # Define the bounding box coordinates
            x1, y1, x2, y2 = int(arr_l[i][0]*1280), int(arr_l[i][1]*720), int(arr_l[i][2]*1280), int(arr_l[i][3]*720)
            
            #x1, y1, x2, y2 = 100, 100, 200, 200

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2),(255, 0 , 0), 2)

        # Save the image with its drawn bounding boxes to the folder labeled 'Marked_Images'

        save_path = 'Marked_Images/'
        name = key + '.jpg'

        # Save the image to the directory
        cv2.imwrite(save_path + name, image)


def main():
    
    print("Bounding Box Analaysis Start Script")

    # Calling function to calculate IOU_Matrixes for each image

    IOU_mat_list, gt_dict, l_dict = IOUMatrix.main()

    # Producing dictionary of false positive and true positive detections for all images

    false_Detect_Dict, true_detection_dict = falseDetectionsAnalysis(IOU_mat_list, gt_dict, l_dict)

    # Graphing a histogram with resolution of localizer bounding boxes on x-axis and frequency of false positive detections on y-axis
    false_detections = plot_Detections(false_Detect_Dict, true_detection_dict)
    
    image_list = list(gt_dict.keys())

    # Drawing bounding boxes on all images from folder 'Check_Files' and placing edited images in folder 'Marked Images'
    drawBoundingBoxes(IOU_mat_list, false_Detect_Dict, gt_dict, l_dict, image_list)

    return false_Detect_Dict, true_detection_dict



if __name__ == '__main__':
    main()