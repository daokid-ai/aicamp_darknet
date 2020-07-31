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
    prefix_path = config.get("PrepData","prefix_path")
    path_to_data = config.get("PrepData", "path_to_data")
    prefix_filenm = config.get("PrepData","prefix_filenm")
    
    return lbl_bxfile, prefix_path, path_to_data, prefix_filenm 


# download image and return image file name
def dl_img(path_to_data, prefix_filenm, image_url, idx):
    try:
        print(f"Downloading: {image_url}")
        response_img = requests.get(image_url)
        if not response_img:
            return "skip"
        
        img_filenm = prefix_filenm + str(idx).zfill(5) + ".jpg"
        with open(os.path.join(path_to_data, img_filenm),'wb') as img_file:
            img_file.write(response_img.content)
            print(f"Saved image at this location: {os.path.join(path_to_data, img_filenm)}")
        return img_filenm
    except:
        return "skip"


# get image from local and find out the dimensions
def get_img_dims(path_to_data, img_filenm):
    width, height = 0, 0
    with Image.open(os.path.join(path_to_data, img_filenm)) as image: 
        width, height = image.size 
    return {"img_width": width, "img_height": height}


# convert 1 label from coordinates into yolo format.
def get_yolo_label(a_box, img_dims):
    dw = 1./img_dims.get("img_width")
    dh = 1./img_dims.get("img_height")
    center_x = dw * (a_box['left'] + a_box['width'])/2.0
    #print(f"Left position: {a_box['left']} + label width: {a_box['width']} = {a_box['left'] + a_box['width']} ==> center_x :{(a_box['left'] + a_box['width'])/2.0}" )
    center_y = dh * (a_box['top'] + a_box['height'])/2.0
    #print(f"Top position: {a_box['top']} + label height: {a_box['height']} = {a_box['top'] + a_box['height']} ==> center_y :{(a_box['top'] + a_box['height'])/2.0}" )
    rel_width = dw * a_box['width']  
    rel_height = dh * a_box['height']

    yolo_list = [center_x, center_y, rel_width, rel_height]
    
    return yolo_list


# write labels in yolo format for 1 image
def write_labels(label_info, lblbxdict_key, object_list, text, img_dims):

    obj_count = 0
    labels = []
    for lbl_dict in label_info.get(lblbxdict_key):
        obj_count += 1
        '''
        if lbl_dict['title'] in object_list:
            print(f"This object is a {object_list[lbl_dict['title']]}")
        else:
            print(f"Uknown object labeled. Check object list if it needs updating.")
        '''
        classification = lbl_dict.get("title")
        class_num = object_list.get(classification)
        a_box = lbl_dict.get("bbox")
        #yolo_info = get_yolo_label(**a_box, **img_dims) use if breaking out values
        yolo_info = get_yolo_label(a_box, img_dims)
        line_to_write = f"{class_num} {' '.join(str(num) for num in yolo_info)}\n"
        text.write(line_to_write)
        print(f"Wrote {line_to_write} to {text.name}")

    print(f"Total of {obj_count} labels written to {text.name}")


def main():
    
    # Setup configuration paths
    prop_file = 'dataprep.properties'
    lbl_bxfile, prefix_path, path_to_data, prefix_filenm = set_configuration(prop_file)
    lblbxdict_key = 'objects'

    # List the different categories in the labels here
    object_list = {'euro': 0, 'dollar' : 1, 'renminbi' : 2, 'pound' : 3 }

    # Read in Labels downloaded from labelbox
    df = pd.read_csv(lbl_bxfile)

    total_rows = len(df.index) 
    print(f"There are a total of {total_rows} images to process.")
    #total_rows = 500
    
    for idx in range(0, total_rows):
        
        json_acceptable_string = df.loc[idx, "Label"]
        
        label_info = json.loads(json_acceptable_string)        
        if not label_info: # the Label has only {}
            print(f"EXCEPTION: There is no label for this image @ index: {idx} ==> Skipping training image.")
            continue
 
        # download image and get width and height
        image_url = df.loc[idx, "Labeled Data"]
        img_filenm = dl_img(path_to_data, prefix_filenm, image_url, idx)
        
        if img_filenm == 'skip':
            print("Error encountered when downloading image from labelbox. Moving on to next image....")
            continue

        img_dims = get_img_dims(path_to_data, img_filenm)    
              
        # create label.txt file for a image
        img_txtnm = prefix_filenm + str(idx).zfill(5) + ".txt"

        with open(os.path.join(path_to_data, img_txtnm), "w") as text:
            if lblbxdict_key in label_info:
                write_labels(label_info, lblbxdict_key, object_list, text, img_dims)
            else:
                print(f"Did not write any yolo values. Empty txt file.")

        print(f"SUCCESS: Training data @ index #: {idx} has been prepared.\n")
        

if __name__ == '__main__':
    try:
        main()
        print('========================End of program===============================')
    except Exception as e:
        raise e("Unknown error in preparing training data.")
        traceback.print_exc()