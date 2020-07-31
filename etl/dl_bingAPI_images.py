import configparser
import os
import errno
import re
import requests
import json
import traceback
from PIL import Image
from io import BytesIO
from requests import exceptions

EXCEPTIONS = set([IOError, FileNotFoundError, OSError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])

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
            hidden_key = re.sub('[a-z,0-9]', "*", lines[0])
            print(f"Retrieved API Key: {hidden_key}")
            
            return lines[0]


# get list of img urls from bing API
def get_list_of_img_urls(search_item, search_url, headers, max_images):
    img_urls = []
    img_index = 0
    while img_index <= max_images:
        # request to the bing search api, the max imgs per request is 150 so it is looped
        params = {'q': search_item, 'count': 150, 'imageType': 'photo', 'offset': img_index}

        r = requests.get( search_url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        search_results = r.json()
        for img in search_results["value"]:
            img_urls.append(img["contentUrl"])
        img_index += 150
    return img_urls


# save a list of images from image urls
def save_list_of_images(img_urls, folder_name):
    for i, contentUrl in enumerate(img_urls):
        try:
            image_data = requests.get(contentUrl, timeout=10)
            image = Image.open(BytesIO(image_data.content))
        except Exception as e:
             # check to see if our exception is in our list of exceptions to check for
            if type(e) in EXCEPTIONS:
                print(f"[INFO] skipping: {contentUrl}")
                continue
            print(f"Unhandled exception while opening image. Exception: {e}")
        ext = image.format
        if ext == 'JPEG': ext =' jpg'
        img_filenm = str(i).zfill(5) + f".{ext}"
        image.save(os.path.join(folder_name, img_filenm), image.format)
        print(f"{i} downloaded from {contentUrl} to {img_filenm}") 

if __name__ == "__main__":

    # config - edit these to your liking
    prop_file = 'dataprep.properties'
    apikey_file, search_url, folder_name, max_images = set_configuration(prop_file)
   
    #Setup initial parameters for search
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
    # get list of images
    img_urls = get_list_of_img_urls(search_item, search_url, headers, max_images)

    # Save list of images
    save_list_of_images(img_urls, folder_name)
    print(f"----------------------- End of program execution --------------------------")