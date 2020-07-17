import configparser
import os
import errno
import requests
import json
import traceback
from PIL import Image
from io import BytesIO


# Read in configurations from properties file
def set_configuration(prop_file):
    if not os.path.exists(prop_file):
        raise Exception("Cannot find properties file.")
    config = configparser.ConfigParser()
    config.read(prop_file)     
    
    #Set configurations
    apikey_file = config.get("BingSearch","key_file_path")
    search_url = config.get("BingSearch","search_url")
    folder_name = config.get("BingSearch","folder_name")
    max_imgs = int(config.get("BingSearch","max_imgs"))
    return apikey_file, search_url, folder_name, max_imgs


# Read bing API Key
def readkey_fromfile(keypath):
    with open(keypath) as keyfile: 
        lines = keyfile.readlines() 
        if len(lines) < 1:
            raise Exception ("The key file is empty.")
        elif len(lines) > 1:
            raise Exception ("Key file may be incorrectly formatted.")
        else:
            print(f"Retrieved API Key: {lines[0]}")
            return lines[0]


if __name__ == "__main__":

    # config - edit these to your liking
    prop_file = 'dataprep.properties'

    apikey_file, search_url, folder_name, max_images = set_configuration(prop_file)
    subscription_key = readkey_fromfile(apikey_file)
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    search_item = 'masks'
    
    # making the folder to store the images
    if not os.path.exists(os.path.dirname(folder_name)):
        try:
            os.makedirs(os.path.dirname(folder_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    img_index = 0
    img_urls = list()
    while img_index <= max_images:
        # request to the bing search api, the max imgs per request is 150 so it is looped
        params = {'q': search_item, 'count': 150, 'imageType': 'photo', 'offset': img_index}
        r = requests.get( search_url, headers=headers, params=params)
        r.raise_for_status()
        search_results = r.json()
        for img in search_results["value"]:
            img_urls.append(img["contentUrl"])
        img_index += 150

    # downloading each image
    for i,item in enumerate(img_urls):
        image_data = requests.get(item)
        try:
            image = Image.open(BytesIO(image_data.content))
        except OSError:
            continue
        image.save(os.path.join(folder_name, str(i) + ".jpg"), image.format)
        print('%d downloaded from %s' % (i, item))