# Libraries
import configparser
import os
import shutil
import traceback


# Read in configurations from properties file
def set_configuration(prop_file):
    if not os.path.exists(prop_file):
        raise Exception("Cannot find properties file.")

    config = configparser.ConfigParser()
    config.read(prop_file)     
    #Set configurations
    prefix_path = config.get("TrainValidTest", "prefix_path")
    train_file = config.get("TrainValidTest", "train_file")
    valid_file = config.get("TrainValidTest", "valid_file")
    test_file = config.get("TrainValidTest", "test_file")
    path_to_train = config.get("TrainValidTest", "path_to_train")
    path_to_valid = config.get("TrainValidTest", "path_to_valid")
    path_to_test = config.get("TrainValidTest", "path_to_test")
    relative_trainpath = config.get("TrainValidTest", "relative_trainpath")
    relative_validpath = config.get("TrainValidTest", "relative_validpath")
    relative_testpath = config.get("TrainValidTest", "relative_testpath")

    return prefix_path, train_file, valid_file, test_file, path_to_train, path_to_valid, path_to_test, relative_trainpath, relative_validpath, relative_testpath


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
def create_img_list(prefix_path, path_to_data, relative_path, file_nm):
    list_jpgs = sorted(img_file for img_file in os.listdir(path_to_data) if img_file.endswith(".jpg"))
    list_jpgs = [os.path.join(relative_path, img_file) for img_file in list_jpgs]
    with open(os.path.join(prefix_path, file_nm), "w") as f:
        f.write("\n".join(list_jpgs))


def main():
    # Setup configuration paths
    prop_file = 'dataprep.properties'
    # Read in configurations from properties file
    prefix_path, train_file, valid_file, test_file, path_to_train, path_to_valid, path_to_test, relative_trainpath, relative_validpath, relative_testpath = set_configuration(prop_file)

    # 80:10:10 :: train:valid:test
    train_files, valid_files, test_files = split_datalist(path_to_train)
    
    # move only the valid and test files into their own folders
    transfer_data(path_to_train, path_to_valid, valid_files)
    transfer_data(path_to_train, path_to_test, test_files)
    
    # Create train.txt, valid.txt, test.txt
    create_img_list(prefix_path, path_to_train, relative_trainpath, train_file)
    create_img_list(prefix_path, path_to_valid, relative_validpath, valid_file)
    create_img_list(prefix_path, path_to_test, relative_testpath, test_file)


if __name__ == '__main__':
    try:
        main()
        print('========================End of program===============================')
    except Exception as e:
        raise e("Unknown error in preparing training data.")
        traceback.print_exc()