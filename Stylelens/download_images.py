#pip install gdown -U --no-cache-dir

import os
import zipfile
import gdown

images_url = "https://drive.google.com/uc?id=1SlE6hs3mtc9Kh17gauHuWfELeUOMojPD&export=download"

dataset_base_path = '.'

zip_file_path = os.path.join(os.path.abspath(dataset_base_path), 'img.zip')

if not os.path.exists(zip_file_path):
   gdown.download(images_url, zip_file_path, quiet=False)

if not os.path.exists(os.path.join(os.path.abspath(dataset_base_path), "img")):
   with zipfile.ZipFile(zip_file_path,"r") as zip_ref:
       zip_ref.extractall(dataset_base_path)