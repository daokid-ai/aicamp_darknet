# Libraries
import configparser
import os
import cv2
import requests
import traceback
from requests import exceptions


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
    path_name = config.get("BingSearch","path_name")
    max_imgs = int(config.get("BingSearch","max_imgs"))
    group_size = config.get("BingSearch","group_size")
    return apikey_file, search_url, folder_name, path_name, max_imgs, group_size


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


def save_images(image_set, total):

    EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])

    for v in image_set:
        
        try:
            # make a request to download the image
            r = requests.get(v["contentUrl"], timeout=30)

            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([path_name, f"{str(total).zfill(8)}{ext}"])
            # save the image
            with open(p,'wb') as f:
                f.write(r.content)
                print(f"Saved image at this location: {p}")

        # catch any errors that would not unable us to download the image
        except Exception as e:
            # check to see if our exception is in our list of exceptions to check for
            if type(e) in EXCEPTIONS:
                print(f"[INFO] skipping: {v['contentUrl']}")
                continue
        
        # check and delete invalid images
        image = cv2.imread(p)
        if image is None:
            print(f"[INFO] deleting: {p}")
            os.remove(p)
            continue

        total += 1
    return total


if __name__ == "__main__":

     # config - edit these to your liking
    prop_file = 'dataprep.properties'
    
    search_term = 'people in stores'

    try:

        apikey_file, URL, folder_name, path_name, MAX_RESULTS , GROUP_SIZE = set_configuration(prop_file)
        subscription_key = readkey_fromfile(apikey_file)
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        if not os.path.exists(os.path.dirname(folder_name)):
            try:
                os.makedirs(os.path.dirname(folder_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise  

        params = {"q": search_term, "offset": 0, "count": GROUP_SIZE}
        search = requests.get(URL, headers=headers, params=params)
        results = search.json()

        if 'error' in results:
            if 'code' in results['error'] :
                raise Exception(f"Error code {results['error']['code']} : {results['error']['message']} ")
        else:
            print(f"These are the results: {results}")
        
        estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)

        print(f"[INFO] {estNumResults} total results for '{search_term}'")
        total = 0

        for offset in range(0, estNumResults, GROUP_SIZE):

            # update the search parameters using the current offset, then make the request to fetch the results
            print(f"[INFO] making request for group {offset}-{offset+GROUP_SIZE} of {estNumResults}...")
            params["offset"] = offset
            search = requests.get(URL, headers=headers, params=params)
            search.raise_for_status()
            results = search.json()
            
            print(f"[INFO] saving images for group {offset}-{offset+GROUP_SIZE} of {estNumResults}...")
            total = save_images(results["value"], total)

    except Exception as e:
        raise e("Unknown error in grabbing images.")
        traceback.print_exc()