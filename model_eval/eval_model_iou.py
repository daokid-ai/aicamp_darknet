# Import libraries
import configparser
import cv2
import numpy as np
import ntpath
import os
import pandas as pd
import pickle
import time
import traceback
# Plotting Libraries
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from matplotlib import rcParams
# Custom lib
from ai import yolo_forward, get_yolo_net, yolo_save_img, yolo_pred_list


rcParams.update({'figure.autolayout': True})
np.set_printoptions(precision=2)
LABELS = ['Euro', 'Dollar', 'Renminbi', 'Pound', 'no detection']
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

def set_configuration(prop_file, section_name):
    '''
    Read in configurations from properties file
    '''
    if not os.path.exists(prop_file):
        raise Exception("Cannot find properties file.")

    try:
        config = configparser.RawConfigParser()
        config.read(prop_file)     
        config_dict = dict(config.items(section_name))
        return config_dict
    except:
        raise Exception("Could not read properties file.")


def get_labels(obj_names_file):
    '''
    Read Labels from obj.names
    '''
    with open(obj_names_file) as objn: 
        labels = objn.read().splitlines() 
        if len(labels) < 1:
            raise Exception ("The obj.names file is empty.")
        else:
            lbl_list = []
            for label in labels:
                lbl_list.append(label)
            lbl_list.append('no detection')
            return lbl_list


def yolo_to_standard_dims(box, H, W):
    """
    Converts YOLO format dimensions into standardized dimensions
    Input:
        box: [relative_center_x, relative_center_y, relative_width, relative_height]
        H: height of image
        W: width of image
    Output: (left_x, top_y, width, height)
    """
    box *= np.array([W, H, W, H])
    center_x, center_y, width, height = box.astype("int")
    x = int(center_x - (width / 2))
    y = int(center_y - (height / 2))

    return (x, y, int(width), int(height))


def get_iou(bb1, bb2):
    """
    Takes the dimensions of two bounding boxes and
    calculates the intersection over union.
    The intersection is the area of overlap between boxes.
    The union is the total area of both boxes.
    The IoU is a float between 0.0 and 1.0
    """
    # Get coordinates of top-left and bottom-right corners
    bb1_x1, bb1_y1, bb1_w, bb1_h = bb1
    bb1_x2, bb1_y2 = bb1_x1 + bb1_w, bb1_y1 + bb1_h
    bb2_x1, bb2_y1, bb2_w, bb2_h = bb2
    bb2_x2, bb2_y2 = bb2_x1 + bb2_w, bb2_y1 + bb2_h

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    # If the boxes do not overlap, return 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bb1_area = bb1_w * bb1_h
    bb2_area = bb2_w * bb2_h

    # Calculates the IoU as the Intersection divided by the Union
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    return iou


def get_prediction(image_folder_root, LABELS, cfg_path, weight_path, confidence_level=0.5):
    """
    Run every image in a given folder through darknet.
    Return a list of dictionaries with pertinent information
    about each image: image_path, class_ids, labels, boxes,
    true_labels, and true_boxes.
    """
    # Loads YOLO into Python
    net = get_yolo_net(cfg_path, weight_path)

    # np.random.seed(42)
    # colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    # Lists all files in the directory and splits them into two lists:
    # one for images and one for txt files
    files = os.listdir(image_folder_root)
    image_paths = sorted([os.path.join(image_folder_root, f) for f in files if '.jpg' in f])
    txt_paths = sorted([os.path.join(image_folder_root, f) for f in files if '.txt' in f])

    # Loops over each image and txt file in the directory
    results = []
    for image_path, txt_path in zip(image_paths, txt_paths):
        try:
            # Get image height and width
            image = cv2.imread(image_path) 
            (H, W) = image.shape[:2]
          
            # Get darknet prediction data
            class_ids, labels, boxes, confidences = yolo_forward(net, LABELS, image, confidence_level)
            print(f"SUCCESS: Predicted Class IDs: {class_ids}; Labels: {labels}; for image: {image_path}.\n")
        except Exception:  # Catch occasional errors
            print(f"ERROR: This image had an error: {image_path}")
            continue

        # Reads ground truth data from txt file
        with open(txt_path, "r") as f:
            txt_labels = f.readlines()

        # Splits data into two lists: labels and boxes
        true_labels = [int(label.split()[0]) for label in txt_labels]
        true_boxes = [label.split()[1:] for label in txt_labels]

        # Convert boxes from YOLO dimensions to standard dimensions
        for i, box in enumerate(true_boxes):
            box = [float(num) for num in box]
            true_boxes[i] = yolo_to_standard_dims(box, H, W)

        # Adds pertinent information to a dictionary and adds the dictionary to the return list
        result = {
            'image_path': image_path,
            'class_ids': class_ids,
            'labels': labels,
            'boxes': boxes,
            'confidences': confidences,
            'true_labels': true_labels,
            'true_boxes': true_boxes
        }
        results.append(result)

    return results    

def clean_scores(results, iou_threshold=0.5):
    '''
    Cleans the results into a simpler list of dictionaries that each contain
    the true value ('true'), the prediction ('pred'), and the confidence
    ('score').
    Checks IoUs to determine whether a prediction is a TP, FP, or FN (TNs are
    not considered)
    '''
    # Loops over every image in the results
    data = []
    for image in results:
        # Extracts data from the dictionary
        class_ids = image.get("class_ids")
        boxes = image.get("boxes")
        confidences = image.get("confidences")
        true_labels = image.get("true_labels")
        true_boxes = image.get("true_boxes")

        # predicted_truths lists the index of the predicted value in relation to
        # the list of true values for an image.
        # It does NOT list the lables.
        # If the model is perfect, this list should be a list of number from 0 to n
        # where n is the number of true boxes on an image
        predicted_truths = []
        # Loops over each predicted box in the image
        for class_id, box, confidence in zip(class_ids, boxes, confidences):
            truths = []
            ious = []
            # Loops over every true box in the image
            # Lists the IOUs and true values of every true lable
            for true_label, true_box in zip(true_labels, true_boxes):
                iou = get_iou(box, true_box)
                ious.append(iou)
                truths.append(true_label)

            # If a prediction doesn't match any true values, it is considered an FP
            best_pred = max(ious)
            if best_pred < iou_threshold:
                data.append({
                    'truth': 4,
                    'pred': class_id,
                    'score': confidence
                })

            else:  # If the prediction matches a true value
                predicted_truth = ious.index(best_pred)
                predicted_truths.append(predicted_truth)

                # If the prediction wasn't a duplicate, add it to the return list
                if len(predicted_truths) == len(set(predicted_truths)):
                    data.append({
                        'truth': truths[predicted_truth],
                        'pred': class_id,
                        'score': confidence
                    })

                else:  # Otherwise, add it as a false positive
                    data.append({
                        'truth': 4,
                        'pred': class_id,
                        'score': confidence
                    })
                    predicted_truths.pop()

        # If the model missed a box, count it a a false negative
        if len(predicted_truths) != len(true_labels):
            for i in range(len(true_labels)):
                if i not in predicted_truths:
                    data.append({
                        'truth': true_labels[i],
                        'pred': 4,
                        'score': 0
                    })

    return data


def get_conf_matrix(data, normalize):
    '''
    Convert the data into a normalized confusion matrix.
    '''
    y_true = [label.get('truth') for label in data]
    y_pred = [label.get('pred') for label in data]

    return confusion_matrix(y_true, y_pred, normalize=normalize)


def display_conf_matrix(conf_matrix, label_list):
    """
    Use matplotlib to display the confusion matrix.
    """
    print(f"label_list = {label_list}")
    matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=label_list)
    matrix_display.plot()
    plt.show()


def plot_confusion_matrix(cm, classes, title='confusion matrix',
                          cmap=plt.cm.rainbow, output_file='confusion_matrix.jpg'):
    '''take confusion matrix, plot it, and save it to file
    '''
    print(type(cm[0, 0]))
    if isinstance(cm[0, 0], np.integer):
        fmt = 'd'
    else:
        fmt = '.2f'

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black", size=12)
    fig.tight_layout()
    plt.autoscale()
    plt.savefig(output_file, bbox_inches="tight")
    return ax



def evaluate_model(label_list, image_folder_root, cfg_path, weight_path, iou_threshold, confidence_level):
    # calls all the other functions
    results = get_prediction(image_folder_root, label_list, cfg_path, weight_path, confidence_level)
    data = clean_scores(results, iou_threshold)
    
    # get confusion matrix not normalized
    conf_matrix = get_conf_matrix(data, normalize = None)
    #display_conf_matrix(conf_matrix, label_list)
    plot_confusion_matrix(conf_matrix, label_list, title='confusion matrix without nomalizaton',
                              cmap=plt.cm.rainbow, output_file='confusion_matrix.jpg')

    # get confusion matrix normalized
    conf_matrix = get_conf_matrix(data, normalize = 'true')
    #display_conf_matrix(conf_matrix, label_list)
    plot_confusion_matrix(conf_matrix, label_list, title='normalized confusion matrix',
                              cmap=plt.cm.rainbow, output_file='normalized_confusion_matrix.jpg')

if __name__ == "__main__":
    try:
        # Load configuration file 
        prop_file = "model_eval.properties"
        section_name = "ModelEval"
        cfg = set_configuration(prop_file, section_name)
        # grab labels from obj.names
        label_list = get_labels(cfg.get('names_path'))
        
        evaluate_model( label_list,
                        cfg.get('path_to_test'),
                        cfg.get('config_path'),
                        cfg.get('weights_path'),
                        float(cfg.get('iou_threshold')),
                        float(cfg.get('confidence_lvl'))
                        )
        
    except Exception as e:
        raise e("Unknown error in evaluating model.")
        traceback.print_exc()