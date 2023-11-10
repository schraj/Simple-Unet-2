import os
import src.data.constants as c

def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ('test.txt' not in file):
              file_list.append(os.path.join(root, file))
    return file_list

def get_file_name(directory, starting_letter):
  files = get_file_list(directory)  
  for file in files:
    base_file = os.path.basename(file)
    if base_file.startswith(starting_letter) and not '_mask' in base_file:
      return base_file
  return None

def get_test_files():
  files = get_file_list(c.SEGMENTATION_TEST_DIR)  
  filtered = []
  for file in files:
    if not '_mask' in file:
      filtered.append(file)
  return filtered

def get_test_inputs_and_targets():
  files = get_test_files()  
  targets = []
  for file in files:
      base_name = os.path.basename(file)
      base_mask_name = base_name.replace('.png', '_mask.png')
      targets.append(os.path.join(c.SEGMENTATION_TEST_DIR, base_mask_name)) 
  return files, targets
