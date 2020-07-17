# Libraries
import configparser
import os
import requests
import json
import pandas as pd
import traceback
# LabelBox Libraries
from labelbox import Client
from labelbox import Project
from labelbox.exceptions import ResourceNotFoundError


# Read in configurations from properties file
def set_configuration(prop_file):
    if not os.path.exists(prop_file):
        raise Exception("Cannot find properties file.")
    config = configparser.ConfigParser()
    config.read(prop_file)     
    #Set configurations
    apikey_file = config.get("LabelBox","key_file_path")
    proj_nm = config.get("LabelBox","my_proj")
    csv_filenm = config.get("LabelBox","csv_file")
    return apikey_file, proj_nm, csv_filenm

# Read LabelBox API Key
def readkey_fromfile(keypath):
    with open(keypath) as keyfile: 
        lines = keyfile.readlines() 
        if len(lines) < 1:
            raise Exception ("The key file is empty.")
        elif len(lines) > 1:
            raise Exception ("Key file may be incorrectly formatted.")
        else:
            return lines[0]

# Save Json to CSV file
def save_json_to_csv(json_url, csv_filenm):
    response = requests.get(json_url)
    if response.status_code == 200 :
        lblbx_df = pd.DataFrame(response.json())
        lblbx_df.to_csv(csv_filenm)
        print("Labelbox data saved to {}".format(csv_filenm))
    else:
        print("Unknown issue. Turn on debugger.")

# Retrieve project from Labelbox
def get_lblbx_proj(projects):
    proj_list = []

    for proj in projects:
        proj_list.append(proj)

    if len(proj_list) == 0:
        raise Exception("There are no projects assigned to user ")
    elif len(proj_list) > 1:
        print("There is more than one project assigned to user. Program will take the first project.")
    else:
        print("There is one project assigned to user.")
    
    return proj_list


if __name__ == '__main__':

    prop_file = 'dataprep.properties'

    try:
        key_path, my_proj, csv_filenm = set_configuration(prop_file)

        LBLBX_API_KEY = readkey_fromfile(key_path)
        client = Client(LBLBX_API_KEY)
        projects_x = client.get_projects(where=Project.name == my_proj)
        
        my_proj_list = get_lblbx_proj(projects_x)
        my_json_url = client.get_project(my_proj_list[0].uid).export_labels()
        save_json_to_csv(my_json_url, csv_filenm)

    except Exception as e:
        raise e("Unknown error in obtaining Labelbox project data set ")
        traceback.print_exc()