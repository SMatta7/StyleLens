from sentence_transformers import SentenceTransformer
from lshashpy3 import LSHash
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import config as c
import sldataset as dataset

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a function to generate text embeddings from image descriptions
def generate_text_embedding(image_description):
    # Generate embedding for the image description
    text_embedding = model.encode(image_description)
    return text_embedding  # Convert to list for Milvus compatibility


if __name__ == '__main__':
  
    train_dataset = dataset.SLDataset(image_filename=c.train_image_filename, caption_filename=c.train_caption_filename, img_dir=c.img_dir, transform='transform')
    #train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)  

    print('extracted image path')

    model = SentenceTransformer('clip-ViT-B-32')

    styleLens_embeddings = []
    img_path_list = []

    count = 0
    # Iterate over each image-description pair in the dataset
    for bath in tqdm(train_dataset):
            try:
                # print(bath['img_path'])
                # print(bath['caption'])
                # count += 1
                # if count > 3:
                #     break
                # print('modle loaded ')
                #img_emb = model.encode(Image.open(img_path), convert_to_tensor=True)   
                img_emb = model.encode(bath['caption'], convert_to_tensor=True)   
                styleLens_embeddings.append(img_emb)      
                img_path_list.append(bath['img_path'])
            except Exception as e:
                print(str(e))

    print('Encoded images')
    combine_image_embeddings = dict(zip(img_path_list, styleLens_embeddings))
    # save to pickle file for the app
    print("Saving text embeddings")
    embedding_path = os.path.join('output', 'sl_text_embedding.pkl')
    with open(embedding_path,'wb') as f:
        pickle.dump(combine_image_embeddings,f)
