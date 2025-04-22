import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

import config as c
import sldataset as dataset
import resnet50 as Resnet50

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs, device, save_path, checkpoint_interval, val_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0    
        for i, batch in tqdm(enumerate(train_loader)):

            inputs, labels = batch['image'].to(c.device), batch['target_label'].to(c.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if (i + 1) % checkpoint_interval == 0:
                torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}_batch_{i+1}.pth')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels = batch['image'].to(c.device), batch['target_label'].to(c.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                # Get the predicted class with the highest probability for each image
                _, predicted = torch.max(outputs, 1) #outputs.data
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        #val_accuracy = correct / total  #*100
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)

# Obtain embeddings for the data
def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.numpy())
    return embeddings, labels

# Main function to execute training and embedding extraction
def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    checkpoint_interval = 10000
    save_path = os.path.join('output', 'stylelens_model_v1.pth')

    train_dataset = dataset.SLDataset(image_filename=c.train_image_filename, caption_filename=c.train_caption_filename, img_dir=c.img_dir, transform='transform')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f'train data set prepared {len(train_dataset)}')


    val_dataset = dataset.SLDataset(image_filename=c.val_image_filename, caption_filename=c.val_caption_filename, img_dir=c.img_dir, transform='transform')
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    print(f'val data set prepared {len(val_dataset)}')
    
    print(f'model creating')
    model = Resnet50.Resnet50()
    model = model.to(c.device)
    print(f'model created')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=0.01

    print(f'training start')
    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, c.device, save_path, checkpoint_interval, val_loader)
    print(f'training end')

if __name__ == "__main__":
    main()
    print("done")