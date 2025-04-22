import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
from torchvision import transforms
from lshashpy3 import LSHash
from tqdm import tqdm
import os

import resnet50 as Resnet
import config as c
import sldataset as dataset

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  #
    return image

# Generate ResNet embeddings
def generate_embeddings(image_path, model):
    image = preprocess_image(image_path)
    # Remove the last fully connected layer
    #embedding_model = torch.nn.Sequential(*list(model.children())[:-1])
    embedding_model = model
    # Set the model to evaluation mode
    embedding_model.eval()
    image = image.to(c.device)
    with torch.no_grad():
        embeddings = embedding_model(image)
    return embeddings.cpu().numpy(), image

# Function to load the model
def load_model():
    file_path = os.path.join('output', 'stylelens_model_v1.pth')    
    model = Resnet.Resnet50() 
    model.load_state_dict(torch.load(file_path, c.device), c.device)
    model.to(c.device)
    return model

if __name__ == '__main__':
    model = load_model()
    print('model loaded!!')

    train_dataset = dataset.SLDataset(image_filename=c.train_image_filename, caption_filename=c.train_caption_filename, img_dir=c.img_dir, transform='transform')
    
    k = 6 # hash size
    #L = 17  # number of tables
    d = 256 # Dimension of Feature vector

    image_embedding = LSHash(hash_size=k, input_dim=d)

    count = 0
    styleLens_embeddings = []
    for batch in tqdm(train_dataset):        
        try:
            image = batch['image']
            #print(image)
            # count += 1
            # if count > 5:
            #      break
            path = batch['img_path']
            #print(path)
            styleLens_embeddings, img = generate_embeddings(path, model)
            #print(styleLens_embeddings)
            image_embedding.index(styleLens_embeddings.squeeze().tolist(), extra_data=path)            
        except Exception as e:
            print(" " + str(e))


    # save to pickle file for the app
    print("Saving image embeddings")
    embedding_path = os.path.join('output', 'sl_image_embedding.pkl')
    with open(embedding_path,'wb') as f:
        pickle.dump(image_embedding,f)