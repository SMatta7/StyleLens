import streamlit as st
from PIL import Image
import streamlit as st
import torch
import pickle
import os
import io

from torchvision.models import resnet50
import numpy as np
from torchvision import transforms
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import resnet50 as Resnet
import config as c

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# Function to load the model
def load_model():    
    file_path = os.path.join('output', 'stylelens_model_v1.pth')    
    model = Resnet.Resnet50() #resnet50(weights='DEFAULT')
    #model.fc = nn.Linear(model.fc.in_features, 17)
    model.load_state_dict(torch.load(file_path, c.device), c.device)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(c.device)
    model.eval()
    return model
    

def load_embeddings():
    embedding_path = os.path.join('output', 'sl_image_embedding.pkl')
    with open(embedding_path,'rb') as f:
        return pickle.load(f)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_image(model, row_image):

    image = Image.open(row_image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_t = torch.unsqueeze(transform(image), 0)    
    image1 = batch_t.to(c.device)

    #embedding_model = torch.nn.Sequential(*list(model.children())[:-1]) #2048
    embedding_model = model

    with torch.no_grad():
        embeddings = embedding_model(image1)
    target_embeddings = embeddings.cpu().numpy()
    return target_embeddings

def preprocess_image(image_path):
    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  
    return image


def generate_embeddings(image_path, model):
    image = preprocess_image(image_path)    
    # Remove the last fully connected layer
    embedding_model = model
    # Set the model to evaluation mode
    embedding_model.eval()    
    image = image.to(c.device)    
    with torch.no_grad():
        embeddings = embedding_model(image)
    
    return embeddings.cpu().numpy()

# Function to perform image search
def perform_image_search(image_file, top_number):

    #file = "stylelens_model_v1.pth"
    model = load_model()
    
    print('model loaded!!')
    image_embeddings = load_embeddings()
    target_image_embeddings = load_image(model, image_file)
    print('image embedng done')

    top_n = top_number
    nn = image_embeddings.query(target_image_embeddings.squeeze().tolist(), num_results=top_n, distance_func="cosine")   
    print('matching is start')

    image_matches = []
    for n in nn:
        data, similarity = n
        embedding, im_path = data
        embeddings_np = np.array(embedding)
        embeddings_2d = embeddings_np.reshape(1, -1)
        cos_sim = cosine_similarity1(target_image_embeddings.squeeze().reshape(1, -1), embeddings_2d)
      
        if cos_sim > 0.75:
            image_matches.append(im_path)
    return image_matches

####Text##########
txt_model = SentenceTransformer('clip-ViT-B-32')

def load_text_embeddings():
    embedding_path = os.path.join('output', 'sl_text_embedding.pkl')
    # load embeddings from file
    with open(embedding_path,'rb') as f:
        #contents = pickle.load(f) becomes...
        contents = CPU_Unpickler(f).load()
        return  contents #pickle.load(f, c.device)

def generate_text_embedding(image_description):
    # Generate embedding for the image description
    text_embedding = txt_model.encode(image_description)
    return text_embedding 

def search(query, k, img_path_list, txt_clip_embed):
    query_emb = txt_model.encode(query, convert_to_tensor=True)
    
    hits = util.semantic_search(query_emb, txt_clip_embed, top_k=k)[0]

    results = []        
    print("Query:")
    print(query)
    for hit in hits:
        print(img_path_list[hit['corpus_id']])
        print(hit['score'])
        if hit['score'] > 0.70:
            results.append(img_path_list[hit['corpus_id']])
        #print(IPImage(os.path.join(img_folder, img_names[hit['corpus_id']]), width=200))
    return results

import numpy as np

def cosine_similarity1(X, Y=None):
    
    if Y is None:
        Y = X

    # Normalize the input matrices
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)

    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm

    # Compute the dot product between the normalized matrices
    similarity_matrix = np.dot(X_normalized, Y_normalized.T)

    return similarity_matrix

def main():

    st.sidebar.title("Stylelens")

    #st.sidebar.radio('drops sub-menu', options=['add drops', 'view drops'])
    #st.sidebar.checkbox('special')
    #col1, col2 = st.columns([1,9])
    st.sidebar.header('App Settings')
    top_number = st.sidebar.slider('Number of Search Results', min_value=9, max_value=20)
    #picture_width = st.sidebar.slider('Picture Width', min_value=50, max_value=100)

    st.sidebar.divider() 
    search_text = st.sidebar.text_input("Type Style to Search Image")
    
    st.sidebar.divider() 
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Upload Image File:", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        search_text = None
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image', use_column_width=10)
        
        # Perform image search
        #if st.sidebar.button('Search'):
        with st.spinner('Searching...'):
            results = perform_image_search(uploaded_file, top_number)
            print(results[:5])
            show_results(results)

    ###################
    if search_text is not None:
        uploaded_file = None
        txt_embeddings = load_text_embeddings()

        img_paths = list(txt_embeddings.keys())
        #print(results[:3])
        styleLens_embeddings = list(txt_embeddings.values())

        if search_text is not None and len(search_text) > 0:        
            results = search(search_text, top_number, img_paths, styleLens_embeddings)
            print(results[:3])
            show_results(results)

    ###################        

def show_results(results):
    st.subheader("Search Results")
    if results:
        # Display results with loading effects
        cols = st.columns(3)  # Adjust the number of columns as needed
        for i, result in enumerate(results):
            with cols[i % 3]:
                with st.spinner("Loading..."):
                    # Replace the spinner with the actual image once it's loaded
                    st.image(result, caption='', use_column_width=True)
    else:
        st.write("No results found.")

if __name__ == "__main__":
    main()