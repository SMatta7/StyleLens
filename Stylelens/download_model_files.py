#pip install gdown -U --no-cache-dir

import os
import zipfile
import gdown

model_url = "https://drive.google.com/uc?export=download&id=1w_8p20-0wmX2UCLFnBLsisr1LRlbUDmM"

dataset_base_path = '.'

zip_file_path = os.path.join(os.path.abspath(dataset_base_path), 'output.zip')

if not os.path.exists(zip_file_path):
   gdown.download(model_url, zip_file_path, quiet=False)

if not os.path.exists(os.path.join(os.path.abspath(dataset_base_path), "output")):
   with zipfile.ZipFile(zip_file_path,"r") as zip_ref:
       zip_ref.extractall(dataset_base_path)