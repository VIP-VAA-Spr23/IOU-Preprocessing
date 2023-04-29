import os
import csv
from bs4 import BeautifulSoup
import pdb
#normalizeBoxCoords() - Normalizes coordinates to (0,1) scale
# Inputs:
    # box: Tuple in the format (xmin, ymin, xmax, ymax)
    # width: Int
    # height: Int
# Output:
    # Tuple in the format (x1,y1,x2,y2)
def normalizeBoxCoords(box, width, height):
    
    #normalizes coordinates
    norm_x1 = box[0] / width
    norm_y1 = box[1] / height
    norm_x2 = box[2] / width
    norm_y2 = box[3] / height
    
    return (norm_x1,norm_y1,norm_x2,norm_y2)

#scrapeLabels() - Parses either Pascal 
# Input:
#   folderPath - Path to folder containing labeled data 
# Output:
#   box_dict - Dictionary that maps each file name to a list of normalized (x,y,w,h) tuples
#   corresponding to each box within the file.
def scrapeLabels(folderPath):
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

            bs_data = BeautifulSoup(data,"xml")

            width = int(bs_data.annotation.size.width.string)
            height = int(bs_data.annotation.size.height.string)
            
            bboxes = []

            for object in bs_data.annotation.find_all('object'):
                xmin = int(float(object.bndbox.xmin.string))
                ymin = int(float(object.bndbox.ymin.string))
                xmax = int(float(object.bndbox.xmax.string))
                ymax = int(float(object.bndbox.ymax.string))

                bbox = (xmin, ymin, xmax, ymax)
                bbox_norm = normalizeBoxCoords(bbox,width,height)

                bboxes.append(bbox_norm)
            
            box_dict[fileName] = bboxes

    return box_dict

def generate_gt_dict():
    bigfolderPath_XML = "./Labeled_Files"
    bigfolderPath_Imgs = "./Images_to_Label"

    folders = os.listdir(bigfolderPath_XML)
    emptyImgs = []
    box_dict = { }
    output_dict = { }
    ogImgFolders = os.listdir(bigfolderPath_Imgs)

    for folder in folders:
        if folder != ".DS_Store" :
            folderPath = bigfolderPath_XML + '/' + folder
            output_dict = scrapeLabels(folderPath)
            box_dict  = {**box_dict, **output_dict}

    for folder in ogImgFolders:
        if folder != ".DS_Store" :

            XML_files = os.listdir(bigfolderPath_XML + '/' + folder)
            Img_files = os.listdir(bigfolderPath_Imgs + '/' + folder)

            xmlSet = set()
            imgSet = set()

            for file in XML_files:
                xmlSet.add(file.split('.')[0])

            for file in Img_files:
                imgSet.add(file.split('.')[0])

            
            intersection = xmlSet.intersection(imgSet)
            diff = imgSet - xmlSet

            # print("Before Correction:")
            # print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(XML_files)}   Overlap: {len(intersection)}   Diff: {len(diff)}")
            extra_label_files = xmlSet - intersection
            
            for i in extra_label_files:
                if i in xmlSet:
                    xmlSet.remove(i)
            
            # print("After Correction:")
            # print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(list(xmlSet))}   Overlap: {len(intersection)}   Diff: {len(diff)}")

            if (len(Img_files) == len(list(xmlSet)) + len(diff)):
                for i in diff:
                    emptyImgs.append(i)
    # pdb.set_trace()
    temp_dict = {img : [(0,0,0,0)] for img in emptyImgs}

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