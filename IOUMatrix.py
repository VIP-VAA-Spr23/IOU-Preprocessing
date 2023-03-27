from GroundTruthBoxes import generate_gt_dict
import torch
import torchvision
import os 
import csv
import numpy as np 
import pdb


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


def main():
    #Ground truth dictionary {'filename':[(xmin1,ymin1,xmax1,ymax1),(xmin2,ymin2,xmax2,ymax2),...]}
    gt_dict = generate_gt_dict() 
    
    l_dict = generate_localization_dict()
    l_dict = clean_localization_dict(l_dict)


    diff = set(gt_dict.keys()) - set(l_dict.keys())

    for key in diff:
        gt_dict.pop(key)

    gt_dict = dict(sorted(gt_dict.items()))
    l_dict = dict(sorted(l_dict.items()))

    iou_mat_list = []

    print([ (k,v) for k,v in l_dict.items() if len(v[0]) == 0])
    print([ (k,v) for k,v in l_dict.items() if len(v[0]) == 0])

    # pdb.set_trace()

    for (k,v),(k2,v2) in zip(gt_dict.items(), l_dict.items()):
        if v == []:
            v = [[0,0,0,0]]
        if v2 == []:
            v2 = [[0,0,0,0]]    
        gtBox = torch.tensor(v) 
        lBox = torch.tensor(v2)
        # pdb.set_trace()

        # matrix = torchvision.ops.box_iou(v, v2)
        # try:
        #     matrix = torchvision.ops.box_iou(gtBox, lBox)
        # except:
        #     pdb.set_trace()
        matrix = torchvision.ops.box_iou(gtBox, lBox)
        
        iou_mat_list.append(matrix)

    # pdb.set_trace()
    

if __name__ == '__main__':
    main()