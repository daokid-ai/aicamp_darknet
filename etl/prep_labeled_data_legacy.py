# Libraries
import configparser
import json
import os
import pandas as pd
import requests 
import traceback
from PIL import Image


# Read in configurations from properties file
def set_configuration(prop_file):
    if not os.path.exists(prop_file):
        raise Exception("Cannot find properties file.")
    config = configparser.ConfigParser()
    config.read(prop_file)     
    #Set configurations
    lbl_bxfile = config.get("PrepData","lbl_bxfile")
    path_to_data = config.get("PrepData", "path_to_data")
    prefix_path = config.get("PrepData","prefix_path")
    prefix_filenm = config.get("PrepData","prefix_filenm")

    return lbl_bxfile, path_to_data, prefix_path, prefix_filenm


# download image and return image file name
def dl_img(path_to_train, prefix_filenm, image_url, idx):
    print(f"Downloading: {image_url}")
    response_img = requests.get(image_url)
    
    if not response_img:
        return "skip"
        
    img_filenm = prefix_filenm + str(idx).zfill(5) + ".jpg"

    with open(os.path.join(path_to_train, img_filenm),'wb') as img_file:
        img_file.write(response_img.content)
        print(f"Saved image at this location: {os.path.join(path_to_train, img_filenm)}")
    return img_filenm


# get image from local and find out the dimensions
def get_img_dim(path_to_train, img_filenm):
    image_dimensions = 0, 0

    with Image.open(os.path.join(path_to_train, img_filenm)) as image: 
        image_dimensions = image.size 

    return image_dimensions

# convert 1 label from coordinates into yolo format.
def get_yolo_label(a_box, key, img_dims):
    list_x = []
    list_y = []

    for vertices in a_box[key]:
        list_x.append(vertices.get('x'))
        list_y.append(vertices.get('y'))

    dw = 1./img_dims[0]
    dh = 1./img_dims[1]

    center_x = dw * (max(list_x) + min(list_x))/2.0
    center_y = dh * (max(list_y) + min(list_y))/2.0
    rel_width = dw * (max(list_x) - min(list_x))  
    rel_height = dh * (max(list_y) - min(list_y)) 

    yolo_list = [center_x, center_y, rel_width, rel_height]
    
    return yolo_list


# write labels in yolo format for 1 image
def write_labels(lbl_set, label_type, cat_num, img_dims, text):
    obj_count = 1
    box_key ='geometry'

    for info in lbl_set[label_type]:
        yolo_vals = get_yolo_label(info, box_key, img_dims)
        lbl_line = f"{cat_num} {yolo_vals[0]} {yolo_vals[1]} {yolo_vals[2]} {yolo_vals[3]}\n"
        text.write(lbl_line)
        obj_count += 1
        print(f"Wrote {lbl_line} to {text.name}")

    print(f"Total of {obj_count} labels written to txt file")        


def main():
    
    # Setup configuration paths
    prop_file = 'dataprep.properties'
    lbl_bxfile, path_to_data, prefix_path, prefix_filenm = set_configuration(prop_file)

    # List the different categories in the labels here
    label_type_0 = 'Face_with_masks'
    label_type_1 = 'Face_no_masks'

    # Read in Labels downloaded from labelbox
    df = pd.read_csv(lbl_bxfile)

    total_rows = len(df.index) 
    print(f"There are a total of {total_rows} images to process.")
    total_rows = 50
    
    for idx in range(0, total_rows):
    
        json_acceptable_string = df["Label"][idx].replace("'", "\"")
        
        if( json_acceptable_string == 'Skip'):
            print(f"This is the label here is {json_acceptable_string} . We will skip this label.")
            continue

        label_info = json.loads(json_acceptable_string)

        if not label_info:
            print(f" At Index {idx}, there is no label info")
            continue
        
        # download image and get width and height
        image_url = df["Labeled Data"][idx]
        img_filenm = dl_img(path_to_data, prefix_filenm, image_url, idx)
        
        if img_filenm == 'skip':
            print("Error encountered when downloading image from labelbox. Moving on to next image....")
            continue

        img_dims = get_img_dim(path_to_data, img_filenm)    
          
              
        # create label.txt file for a image
        img_txtnm = prefix_filenm + str(idx).zfill(5) + ".txt"
        text = open(os.path.join(path_to_data, img_txtnm), "w+")

        if label_type_0 in label_info:
            cat_num = 0
            write_labels(label_info, label_type_0, cat_num, img_dims, text)

        if  label_type_1 in label_info:
            cat_num = 1
            write_labels(label_info, label_type_1, cat_num, img_dims, text)

        text.close()

        print(f"Prepared training data @ index #: {idx}")
        

if __name__ == '__main__':
    try:
        main()
        print('========================End of program===============================')
    except Exception as e:
        raise e("Unknown error in preparing training data.")
        traceback.print_exc()