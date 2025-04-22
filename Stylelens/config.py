import torch
import os


img_dir = '.'

app_resources = 'output'

base_dataset_file = os.path.join('dataset', 'preprossed-files')

train_image_filename = os.path.join(base_dataset_file, 'train_img.txt')
train_caption_filename = os.path.join(base_dataset_file, 'train_txt_labels.txt')

val_image_filename =  os.path.join(base_dataset_file, 'query_img.txt')
val_caption_filename =  os.path.join(base_dataset_file, 'query_txt_labels.txt')

gal_image_filename =  os.path.join(base_dataset_file, 'gallery_img.txt')
gal_caption_filename =  os.path.join(base_dataset_file, 'gallery_txt_labels.txt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_size = 256

def get_label():
        image_categories = ['Dresses', 'Skirts', 'Blouses_Shirts', 'Sweatshirts_Hoodies', 'Cardigans',
                            'Jackets_Coats', 'Sweaters', 'Tees_Tanks', 'Shorts', 'Rompers_Jumpsuits',
                            'Graphic_Tees', 'Pants', 'Denim', 'Jackets_Vests', 'Leggings', 'Shirts_Polos',
                            'Suiting']

        category_to_label = {category: i for i, category in enumerate(image_categories)}
        return category_to_label