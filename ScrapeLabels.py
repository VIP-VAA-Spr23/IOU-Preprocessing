import os
import csv
from bs4 import BeautifulSoup

#normalizeBoxCoords() - Normalizes coordinates to (0,1) scale
# Inputs:
    # box: Tuple in the format (xmin, ymin, xmax, ymax)
    # width: Int
    # height: Int
# Output:
    # Tuple in the format (x,y,w,h)
def normalizeBoxCoords(box, width, height):

    #calculates width and height of box
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]

    #calculates center of box
    box_center_x = box[0] + (box_width / 2)
    box_center_y = box[1] + (box_height / 2)

    #normalizes coordinates
    norm_x = box_center_x / width
    norm_y = box_center_y / height
    norm_w = box_width / width
    norm_h = box_height / height

    return (norm_x,norm_y,norm_w,norm_h)

#scrapeLabels() - Parses Pascal VOC XML Files
# Input:
#   folderPath - Path to folder containing labeled data
# Output:
#   box_dict - Dictionary that maps each file name to a list of normalized (x,y,w,h) tuples
#   corresponding to each box within the file.
def scrapeLabels(folderPath):
    fileNames = []
    box_dict = {}
    if folderPath != "./Labeled_Files/.DS_Store" :
        files = os.listdir(folderPath)


        for file in files:
            fileName, fileType = file.split('.') #Seperates files into variables for file name and extension type
            fileNames.append(fileName) #Appends filename to list
            filePath = folderPath + '/' + file

            #If label is in Pascal Format
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

                    bbox = [xmin, ymin, xmax, ymax]
                    bbox_norm = normalizeBoxCoords(bbox,width,height)

                    bboxes.append(bbox_norm)

                box_dict[fileName] = bboxes

            ## doesn't work right now
            #If label is in YOLO format
            # elif (fileType == "txt"):
            #     bboxes = []
            #     with open(filePath) as file:
            #          for line in file:
            #               split = line.split() #splits the line on whitespace
            #               split.pop(0)  #removes the inital 0 at the start of each line
            #               bbox = tuple(map(float, split)) #convert numbers to floats and store them in a tuple
            #               bboxes.append(bbox) #append tuple to list of tuples
            #     box_dict[fileName] = bboxes

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

def main():
    bigfolderPath_XML = "./Labeled_Files"
    emptyImgs = []

    folders = os.listdir(bigfolderPath_XML)

    box_dict = { }
    output_dict = { }

    for folder in folders:
        folderPath = bigfolderPath_XML + '/' + folder
        output_dict = scrapeLabels(folderPath)
        box_dict  = {**box_dict, **output_dict}


    bigfolderPath_Imgs = "./Images_to_Label"

    ogImgFolders = os.listdir(bigfolderPath_Imgs)



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

            extra_label_files = xmlSet - intersection
            print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(XML_files)}   Overlap: {len(intersection)}   Diff: {len(diff)}")

            for i in extra_label_files:
                if i in xmlSet:
                    xmlSet.remove(i)


            print(f"{folder} -- Image Count: {len(Img_files)}   XML Count: {len(list(xmlSet))}   Overlap: {len(intersection)}   Diff: {len(diff)}")

            if (len(Img_files) == len(list(xmlSet)) + len(diff)):
                for i in diff:
                    emptyImgs.append(i)
    
    



    print(emptyImgs)
    print(box_dict)


    # outputFolderPath = "./ground_truth_outputs_csv"
    # generateCSVFiles(box_dict,outputFolderPath)


if __name__ == '__main__':
    main()
