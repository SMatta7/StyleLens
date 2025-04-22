import json
import os
import csv
import pandas as pd
from tqdm import tqdm

RESULT_PATH = 'preprossed-files'

def split_img_dataset():
    print('split img dataset...')
    fn = open('list_eval_partition.txt').readlines()
    train = open(os.path.join(RESULT_PATH, 'train_img.txt'), 'w')
    query = open(os.path.join(RESULT_PATH, 'query_img.txt'), 'w')
    gallery = open(os.path.join(RESULT_PATH, 'gallery_img.txt'), 'w')

    for _, line in tqdm(enumerate(fn[2:])):
        aline = line.strip('\n').split()
        img, _, prefix = aline[0], aline[1], aline[2]

        if prefix == 'train':
            train.write(img)
            train.write('\n')
        else:
            if prefix == 'query':
                query.write(img)
                query.write('\n')

            elif prefix == 'gallery':
                gallery.write(img)
                gallery.write('\n')

    train.close()
    query.close()
    gallery.close()

def extract_image_description():
    print('extract image description...')
    with open("list_description_inshop.json", "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    rows = []
    for item in tqdm(json_data):
        item_id = item['item']
        color = item['color'].replace('-', ' ')
        sentences_with_color = f'color is {color} ' + ' '.join(item['description'][1:3])
        rows.append((item_id, sentences_with_color))

    # Write the rows to a CSV file
    csv_file = os.path.join(RESULT_PATH, 'sentences-with-item.csv')
    with open(csv_file, mode='w',  encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Item', 'Sentence'])  # Write header
        writer.writerows(rows)

def preprocess_caption(line, text2label_df):
    
    aline = line.strip('\n').split('/')
    id = aline[3]    
    label = text2label_df[text2label_df['Item'] == id]['Sentence'].values[0]

    category = aline[2].replace('_', ' ')
    caption = f'{aline[1]} {category} ' + label

    return caption.lower()


def generate_imgage_label():
    print('generateing imgage labels...')
    csv_file = os.path.join(RESULT_PATH, 'sentences-with-item.csv')
    text2label_df = pd.read_csv(csv_file, encoding='utf-8')
 
    def get_label(fn, prefix):
        print(f'labels generating - {prefix}')
        rf = open(fn, encoding='utf-8').readlines()
        with open(os.path.join(RESULT_PATH, '%s_txt_labels.txt' % prefix), 'w', encoding='utf-8') as txtlblfile:
            for line in tqdm(rf):
                caption = preprocess_caption(line, text2label_df)
                txtlblfile.write(caption)
                txtlblfile.write('\n')
            txtlblfile.close()

    get_label(os.path.join(RESULT_PATH, 'train_img.txt'), 'train')
    get_label(os.path.join(RESULT_PATH, 'gallery_img.txt'), 'gallery')
    get_label(os.path.join(RESULT_PATH, 'query_img.txt'), 'query')

if __name__ == '__main__':
    split_img_dataset()
    extract_image_description()
    generate_imgage_label()
