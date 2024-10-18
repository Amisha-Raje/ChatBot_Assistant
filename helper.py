import os
import shutil
import math

def group_files_into_batches(source_folder, batch_size=1):
    # List all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Create batches of files
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        batch_folder = os.path.join(source_folder, f'batch_{i // batch_size + 1}')
        
        # Create a new batch folder
        os.makedirs(batch_folder, exist_ok=True)
        
        # Move files to the new batch folder
        for file in batch_files:
            shutil.move(os.path.join(source_folder, file), os.path.join(batch_folder, file))
        
        print(f'Moved {len(batch_files)} files to {batch_folder}')

# total = 0

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

    
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Check if it is a file
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return (total_size)



def count(parent_folder):
    total=0
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder,folder)
        k = len(os.listdir(folder_path))
        size = get_folder_size(folder_path)
        total+=size
        print(f"{folder_path} - {k}  size = {convert_size(size)}")
    return total
# print(ct)



# Usage
source_folder = './final_files_copy'  # Change this to your folder path
# group_files_into_batches(source_folder, batch_size=40)
# count(source_folder)
# print(f"total size = {convert_size(count(source_folder))}")