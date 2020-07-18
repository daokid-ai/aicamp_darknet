# Libraries
import configparser
import os
import shutil
import traceback


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

#list data from one folder and create three lists based on 80/10/10 split
def split_datalist(path_to_train):
    train_files = sorted(os.listdir(path_to_train))
    # splits into two different lists and zips txt and jpg files
    txt_files = [file for file in train_files if file.endswith(".txt")]
    jpg_files = [file for file in train_files if file.endswith(".jpg")]
     # two dimensional list with only two items per slot
    train_files = list(zip(txt_files, jpg_files))

    # list 10% of the files into the test_files list and another 10% into valid_files
    train_files = [file for i, file in enumerate(train_files) if (i % 10 != 0) and ((i+1) % 10 != 0)]
    valid_files = [file for i, file in enumerate(train_files) if (i+1) % 10 == 0]
    test_files = [file for i, file in enumerate(train_files) if i % 10 == 0]
    
    # flattens the lists & converts lists of lists into single list
    # [("1.txt", "1.jpg"), ("2.txt", "2.jpg"), ("3.txt", "3.jpg")] --> ["1.txt", "1.jpg", "2.txt", "2.jpg", "3.txt", "3.jpg"]
    train_files = [file for pair in train_files for file in pair]
    valid_files = [file for pair in valid_files for file in pair]
    test_files = [file for pair in test_files for file in pair]

    return train_files, valid_files, test_files


# Move data based on split list 
def transfer_data(from_loc, to_loc, list_files):
    for data_file in list_files:
        source = os.path.join(from_loc, data_file)
        destination = os.path.join(to_loc, data_file)
        try:
            shutil.move(source, destination)
        except Exception as e:
            raise e(f"Could not move data from {source} to {destination}") 


# Create text file with  list of images and relative paths.
def create_img_list(prefix_path, path_to_data, prefix_filenm, relative_path, file_nm):
    list_jpgs = sorted(img_file for img_file in os.listdir(path_to_data) if img_file.endswith(".jpg"))
    list_jpgs = [os.path.join(relative_path, img_file) for img_file in list_jpgs]
    with open(os.path.join(prefix_path, prefix_filenm + file_nm), "w") as f:
        f.write("\n".join(list_jpgs))


if __name__ == '__main__':
    try:
        # Setup configuration paths
        prop_file = 'dataprep.properties'
        section_name = "TrainValidTest"
        # Read in configurations from properties file
        cfg = set_configuration(prop_file, section_name)
        
        path_to_train = cfg.get('path_to_train')
        path_to_valid = cfg.get('path_to_valid')
        path_to_test = cfg.get('path_to_test')
        prefix_path = cfg.get('prefix_path')
        prefix_filenm = cfg.get('prefix_filenm')
        # 80:10:10 :: train:valid:test
        train_files, valid_files, test_files = split_datalist(path_to_train)
      
        # move only the valid and test files into their own folders
        transfer_data(path_to_train, path_to_valid, valid_files)
        transfer_data(path_to_train, path_to_test, test_files)
       
        # Create train.txt, valid.txt, test.txt
        create_img_list(prefix_path, path_to_train, prefix_filenm, cfg.get('relative_trainpath'), cfg.get('train_file'))
        create_img_list(prefix_path, path_to_valid, prefix_filenm, cfg.get('relative_validpath'), cfg.get('valid_file'))
        create_img_list(prefix_path, path_to_test, prefix_filenm, cfg.get('relative_testpath'), cfg.get('test_file'))
        print('========================End of program===============================')
    except Exception as e:
        raise e("Unknown error in preparing training data.")
        traceback.print_exc()