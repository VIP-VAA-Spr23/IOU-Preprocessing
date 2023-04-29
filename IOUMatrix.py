from GroundTruthBoxes import generate_gt_dict
import torch
import torchvision
import os 
import csv
import numpy as np 
import pdb
import pandas as pd

'''This script is used to calculate the IoU Matrix for each image in a sample set.
The IoU matrix is the intersection over union between ground truth bounding boxes (gt)
and localizer generated bounding boxes (l). We are essentially trying to calculate how much 
the localizer and ground truth bounding boxes overlap.

Please note that this script will produce a n x m matrix for each image in the sample set, 
where n = number ground truth bounding boxes and m = number of localizer generated bounding boxes.

General Script Steps:

1) Generate Localizer Dictionary 
2) Calculate IoU matrix for each image using that image's ground truth and localizer dictionaries in the iOU function
3) IoU_mat_list in main() function is the list of all the IoU matrixes from the sample set 

Please note that in the main() function, the dictionaries: gt_dict and l_dict are sorted in the same
order as IoU_mat_list so you can easily access the each images IoU_matrix

'''


#Removes ['x', 'y', 'w', 'h'] header from box list for each file and non-animal class rows
#Removes ['x', 'y', 'w', 'h'] header from box list for each file and non-animal class rows
def clean_localization_dict(l_dict):

    new_dict = {}

    for filename,box_list in l_dict.items():

        if len(box_list) == 1:
            tups = []
            tup = (0,0,0,0)
            tups.append(tup)
            new_dict[filename] = tups
        else:
            tups = []
            box_list.pop(0)
            for box in box_list:
                if (box[-1] != '1'):
                    tup = (0,0,0,0)
                else:
                    box.pop(-1)
                    box = [float(x) for x in box]
                    tup = (box[0],box[1],box[0]+box[2],box[1]+box[3])
                tups.append(tup)

            new_dict[filename] = tups
    return new_dict
            
def generate_localization_dict():
    folder_path = "./outputs_csv"

    l_dict = {}

    #Iterates through folder of CSV files and creates dictionary in the format
    # {"filename"}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = []
                for row in reader:
                    rows.append(row)
                l_dict[filename.split('.')[0]] = rows
    return l_dict


def iOU(localizer, ground_truth):

    # Define the two arrays of bounding boxes (in the format [x1, y1, x2, y2])

    # Assigning coodinate names for the bounding boxes
    local_names = []
    ground_names = []

    # Assign names for localizer bounding boxes in the following string format: x1, y1, x2, y2
    for i in range(len(localizer)):
        strCoors = ""

        for j in range(len(localizer[i])):
            strCoors = strCoors + str(localizer[i][j])

            if j != len(localizer[i]) - 1:
                strCoors += ','

        local_names.append(strCoors)
    
    # Assign names for ground truth bounding boxes in the following string format: x1, y1, x2, y2
    for i in range(len(ground_truth)):
        
        strCoors = ""

        for j in range(len(ground_truth[i])):

            strCoors = strCoors + str(ground_truth[i][j]) + ','

            if j != len(ground_truth[i]) - 1:
                strCoors += ','

        ground_names.append(strCoors)
    
    # Calculate the IoU between all pairs of boxes using the bbox_overlaps function
    iou_matrix = torchvision.ops.box_iou(torch.tensor(ground_truth),torch.tensor(localizer))

    # Assign row and column names to dataframe (This is an Optional Inclusion of gt and l names, but unecessary)
    #df = pd.DataFrame(iou_matrix.numpy(), index=ground_names, columns=local_names)

    df = pd.DataFrame(iou_matrix.numpy())

    # df is the IoU matrix for a single image
    return df

def generatetruePositivityArray(bestBoundingBox):

    # This method returns an array of the true positivity ratio within each image of the sample

    # Array to be returned is truePositivity -> length of array is the same as the number of images 
    # Please note that images have been sorted based on number assigned to them (Ex. 1013402340.png)

    truePositivity = []

    for i in range(len(bestBoundingBox)):
        
        # Counter for the number of good localizer bounding boxes with a IoU > 0.5
        goodBoxes = 0  
        totalBoxes = len(bestBoundingBox)

        for j in range(len(bestBoundingBox[i])):
            
            if bestBoundingBox[i][j][3] >= 0.5:
                goodBoxes += 1
        
        # Append ratio of good bounding boxes to total localizer bounding boxes to truePositivity
        truePositivity.append((i, goodBoxes/totalBoxes))
    
    return truePositivity

def main():

    # Main script will produce a list of IoU matrixes where each item is a IoU matrix for 1 image
    # All dictionaries and lists are sorted based on the image number assigned (Ex. 1013402340.png))

    #Ground truth dictionary {'filename':[(xmin1,ymin1,xmax1,ymax1),(xmin2,ymin2,xmax2,ymax2),...]}
    gt_dict = generate_gt_dict() 
    
    #Localizer Dictionary 
    l_dict = generate_localization_dict()
    l_dict = clean_localization_dict(l_dict)


    diff = set(gt_dict.keys()) - set(l_dict.keys())

    for key in diff:
        gt_dict.pop(key)

    # Sorting gt (ground_truth) and l (localizer) dictionaries by image number (Ex. 1013402340.png)
    gt_dict = dict(sorted(gt_dict.items()))
    l_dict = dict(sorted(l_dict.items()))


    # iou_mat_list -> storing the iou matrix for each image
    iou_mat_list = []

    # For loop to deal with cases where an image has no ground truth or localizer bounding boxes
    for (k,v),(k2,v2) in zip(gt_dict.items(), l_dict.items()):
        
        # Assigning dummy values to gtBox and lBox
        if v == []:
            v = [[0,0,0,0]]
        if v2 == []:
            v2 = [[0,0,0,0]]  

        gtBox = v # ground truth bounding boxes for image x
         
        lBox = v2 # localizer bounding boxes for image x

        matrix = iOU(lBox, gtBox) # generate iou matrix for image x
        iou_mat_list.append(matrix) # append image x to iou_mat_list

    return iou_mat_list, gt_dict, l_dict
    

if __name__ == '__main__':
    main()