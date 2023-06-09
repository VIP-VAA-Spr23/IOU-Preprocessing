import os
import csv
from bs4 import BeautifulSoup
import pdb

def normalizeBoxCoords(box, width, height):
    """
    Normalize coordinates to (0,1) scale.

    Args:
        box: Tuple in the format (xmin, ymin, xmax, ymax).
        width: Int representing the width of the image.
        height: Int representing the height of the image.

    Returns:
        Tuple in the format (x1, y1, x2, y2).
    """
    norm_x1 = box[0] / width
    norm_y1 = box[1] / height
    norm_x2 = box[2] / width
    norm_y2 = box[3] / height
    
    return (norm_x1,norm_y1,norm_x2,norm_y2)


def scrapeLabels(folderPath):
    """
    Extract labeled data from XML files and create a dictionary that maps each file name to a list of normalized (x, y, w, h) tuples corresponding to each box within the file.

    Args:
        folderPath: Path to folder containing labeled data.

    Returns:
        Dictionary that maps each file name to a list of normalized (x, y, w, h) tuples corresponding to each box within the file.
    """
    fileNames = []
    files = os.listdir(folderPath)
    
    box_dict = {}

    for file in files:
        fileName, fileType = file.split('.') #Seperates files into variables for file name and extension type
        fileNames.append(fileName) #Appends filename to list
        filePath = folderPath + '/' + file

        if (fileType == "xml"):

            with open(filePath, 'r') as f:
                data = f.read()
            # Parse the xml data using BeautifulSoup
            bs_data = BeautifulSoup(data,"xml")
            # Extract image dimensions
            width = int(bs_data.annotation.size.width.string)
            height = int(bs_data.annotation.size.height.string)
            
            bboxes = []
            # Loop through all objects and extract bounding boxes
            for object in bs_data.annotation.find_all('object'):
                xmin = int(float(object.bndbox.xmin.string))
                ymin = int(float(object.bndbox.ymin.string))
                xmax = int(float(object.bndbox.xmax.string))
                ymax = int(float(object.bndbox.ymax.string))

                bbox = (xmin, ymin, xmax, ymax)
                # Normalize the bounding box coordinates to be between 0 and 1
                bbox_norm = normalizeBoxCoords(bbox,width,height)

                bboxes.append(bbox_norm)
            
            box_dict[fileName] = bboxes

    return box_dict

def generate_gt_dict():
    """
    Generates a dictionary containing bounding boxes for each image file in the "Images_to_Label" folder.
    
    Returns:
    - A dictionary with image filenames as keys and a list of bounding boxes as values.
    """
    # Set the path for the folders containing folders of each person's XML files and images
    bigfolderPath_XML = "./Labeled_Files"
    bigfolderPath_Imgs = "./Images_to_Label"
    # Get a list of all the folders in the XML files folder
    folders = os.listdir(bigfolderPath_XML)
    # Initialize empty lists and dictionaries
    emptyImgs = []
    box_dict = { }
    output_dict = { }
    # Initialize empty lists and dictionaries
    ogImgFolders = os.listdir(bigfolderPath_Imgs)
    # Loop through each folder in the XML files folder
    for folder in folders:
        # Ignore the .DS_Store file
        if folder != ".DS_Store" :
            # Set the path for the current folder
            folderPath = bigfolderPath_XML + '/' + folder
            # Scrape the labels from the XML files in the folder and add them to the box dictionary
            output_dict = scrapeLabels(folderPath)
            box_dict  = {**box_dict, **output_dict}
    
    # Loop through each folder in the images folder
    for folder in ogImgFolders:
        if folder != ".DS_Store" :
            # Get a list of all the XML files and image files in the current folder
            XML_files = os.listdir(bigfolderPath_XML + '/' + folder)
            Img_files = os.listdir(bigfolderPath_Imgs + '/' + folder)
            # Create sets of the names of the XML files and image files
            xmlSet = set()
            imgSet = set()
            # Add the names of the XML files to the XML set
            for file in XML_files:
                xmlSet.add(file.split('.')[0])
            # Add the names of the image files to the image set
            for file in Img_files:
                imgSet.add(file.split('.')[0])

            # Find the intersection and difference between the XML set and image set
            intersection = xmlSet.intersection(imgSet)
            diff = imgSet - xmlSet
            # Print the counts of the files before and after correction (commented out)
            # print("Before Correction:")
            # print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(XML_files)}   Overlap: {len(intersection)}   Diff: {len(diff)}")
            extra_label_files = xmlSet - intersection
            
            for i in extra_label_files:
                if i in xmlSet:
                    xmlSet.remove(i)
            # Print the counts of the files before and after correction (commented out)
            # print("After Correction:")
            # print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(list(xmlSet))}   Overlap: {len(intersection)}   Diff: {len(diff)}")
            
            # If the number of image files is equal to the number of XML files plus the number of extra image files
            if (len(Img_files) == len(list(xmlSet)) + len(diff)):
                # Add the names of the extra image files to the empty image list
                for i in diff:
                    emptyImgs.append(i)
    # Create a temporary dictionary with the names of the empty images as keys and a default bounding box as values
    temp_dict = {img : [(0,0,0,0)] for img in emptyImgs}
    # Add the temporary dictionary to the box dictionary
    box_dict  = {**box_dict, **temp_dict}
    return box_dict
    
def generateCSVFiles(dict,outputFolderPath):
     #iterates through each filename in the dictionary  
    for filename, boxList in dict.items():
        #creates path to a CSV file to write to 
        csv_path = os.path.join(outputFolderPath, f"{filename}.csv")
        
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["x", "y", "w", "h", "class"])
            for box in boxList:
                x, y, w, h = box
                writer.writerow([x, y, w, h, 1])