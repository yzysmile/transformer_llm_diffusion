import os
import argparse
from sklearn.model_selection import train_test_split

def parse_args():

    parser = argparse.ArgumentParser(description="Script for dividing point cloud data into training, validation, and test sets.")
    parser.add_argument('--data-path', default='/home/yzyrobot/learning_data/factory_data/extract/', type=str,
                        help='Path to the data folder.')
    return parser.parse_args()

def check_folder_existence(folder_path):
    """
    Check if a given folder exists.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        bool: True if folder exists, otherwise False.
    """
    return os.path.exists(folder_path)

def collect_data(data_path):
    """
    Collect data file paths from the specified directory and its subdirectories.

    Args:
        data_path (str): Path to the root directory containing the data.

    Returns:
        tuple: A list of data file paths and a list of sample data file paths.
    """
    sample_data_list = []
    
    data_folder_list = os.listdir(data_path)
    for data_folder_name in data_folder_list:
        # Collect sample point files
        sample_folder = os.path.join(data_path, data_folder_name, "sample_points")
        sample_files = [os.path.join(sample_folder, file) for file in os.listdir(sample_folder)]
        sample_data_list.extend(sample_files)
        
    return sample_data_list

def split_data(data_list, train_ratio, test_ratio, valid_ratio):
    """
    Split the data into training, validation, and test sets using specified ratios.

    Args:
        data_list (list): List of data file paths.
        train_ratio (float): Ratio of training data.
        test_ratio (float): Ratio of test data.
        valid_ratio (float): Ratio of validation data.

    Returns:
        tuple: Training data, validation data, and test data.
    """
    # Split into training and temporary (test + validation) data
    train_data, temp_data = train_test_split(data_list, train_size=train_ratio, random_state=42)
    
    # Split temporary data into test and validation sets
    valid_data, test_data = train_test_split(temp_data, test_size=test_ratio / (test_ratio + valid_ratio), random_state=42)
    
    return train_data, valid_data, test_data

def write_to_file(data_path, file_list, txt_name, mode="w"):
    """
    Helper function to write data to a file.

    Parameters:
    data_path (str): 文件存储的目录路径。
    file_list (list): 文件列表。
    txt_name (str): 要写入的txt文件名（包含扩展名，例如 "output.txt"）。
    mode (str): 写入模式，默认为 "w"（覆盖写入）。可选 "a"（追加写入）。
    """
    # 确保目标目录存在
    os.makedirs(data_path, exist_ok=True)

    # 构造完整的文件路径
    file_path = os.path.join(data_path, txt_name)

    try:
        # 打开文件并写入数据
        with open(file_path, mode) as file:
            for path in file_list:
                file.write(path + "\n")
        print(f"数据已成功写入 {file_path}")
    except Exception as e:
        print(f"写入文件失败：{e}")

def main():
    # Parse arguments
    args = parse_args()
    data_path = args.data_path
    
    # Check if the data folder exists
    if not check_folder_existence(data_path):
        print(f"Folder does not exist: {data_path}")
        return
    
    # Collect data file paths
    sample_data_list = collect_data(data_path)
    print(f"Total number of data files: {len(sample_data_list)}")
    
    # Split data into train, validation, and test sets
    train_ratio, test_ratio, valid_ratio = 0.7, 0.1, 0.2
    train_data, valid_data, test_data = split_data(sample_data_list, train_ratio, test_ratio, valid_ratio)
    
    # Output sizes of each dataset
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(valid_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Write data to corresponding text files
    write_to_file(data_path, train_data, "train.txt")
    write_to_file(data_path, valid_data, "valid.txt")
    write_to_file(data_path, test_data, "test.txt")
    
    print("Updated train.txt, valid.txt, and test.txt.")

if __name__ == "__main__":
    main()
