import BoundingBoxAnalysis



def main():
    
    print("IOU Preprocessing Begins")

    print("Part 1 of script - Finding False Positive and True Positive Detections")

    ''' (Part 1 of Script)
        Script generates 2 dictionaries of the false positive detections and true positive detections

        Dictionaries of detections (dictionary key is image name & value is array of bounding boxes):

        false_positive_detection_dict = all false positive detections among all images
        true_detection_dict = true positive detections for all images
        total_detections = counter for '''

    false_positive_detect_dict, true_detection_dict = BoundingBoxAnalysis.main()



if __name__ == "__main__":
    main()