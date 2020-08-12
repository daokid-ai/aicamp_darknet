# Building Baseline aicamp project work

## config folder
Contains previous projects with *.data, *.names, and *.cfg files

## etl folder
Contains python code to collect images using Bing API
Contains python script to grab list of labels created on Labelbox
Contains python script to download image from labelbox and transform label information into yolo format
Contains python script to split labeled data to 80/10/10

## keys folder
Put your keys here if you need to access API with keys 
Bing requires API key to use their search
labelbox API requires key to extract label information as json stream (alternative: download from website)

## model_eval folder
Contains python script to evaluate trained model

## Notes to use code to help train under darknet (draft)
1. git clone repository to under darknet folder where darknet was cloned and compiled
2. configure dataprep.properties to accordingly (details tbd)
3. Run dl_bingAPI_images.py to grab images using Azure cognitive services for images
4. Run dl_labels_tocsv.py to grab label info from Labelbox.com
5. Run prep_labeled_data.py to prepare labeled data and images for yolo format
6. Run prep_stage_data.py to split data into train set, valid set, and test set (80/10/10) 
7. To train configure and Setup your yolov3.cfg, obj.names, and obj.data, and download darknet53.conv.74 weights 
8. Run in command line interface: ./darknet detector train name_of_your_obj.data name_of_your_yolov3.cfg darknet53.conv.74 -dont_show -map
    a. -dont_show hides detailed errors from output
    b. -map uses mean average precision to train and uses valid data set
    c. -clear to train from the beginning
9. After Training, to perform some spot checking use the following command: 
    a. ./darknet detector test your.data your.cfg backup/your_final.weights data_for_test/image_name.jpg 
10. Configure model_eval.properties and run eval_mode_iou.py to view model performance
