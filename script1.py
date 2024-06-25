#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from torchvision import transforms
from google.cloud import storage
from PIL import Image
import io
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import models
import torch.optim as optim
import os
from google.cloud import storage
import tempfile

project_path = 'gs://wtum_files/'



#wczytanie etykiet
df = pd.read_json(project_path + 'train_annotations')
df_valid = pd.read_json(project_path + 'valid_annotations')
df = df[['image_id', 'category_id']]
df_valid = df_valid[['image_id', 'category_id']]


# Funkcja do wczytania plików z Google Cloud Storage
def list_files_in_bucket(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

# Wczytanie ścieżek do plików
Id_train = list_files_in_bucket('wtum_files', 'train')
Id_valid = list_files_in_bucket('wtum_files', 'valid')


train = pd.DataFrame()
train = train.assign(filename = Id_train)
train['image_id'] = train['filename'].str.replace('train/image_id_','')
train['image_id'] = train['image_id'].str.replace('.jpg','')
train = train[~train['image_id'].str.contains('[^0-9]')]  # Zostaw tylko wartości zawierające cyfry
train['image_id'] = train['image_id'].astype(int)

valid = pd.DataFrame()
valid = valid.assign(filename = Id_valid)
valid['image_id'] = valid['filename'].str.replace('valid/image_id_','')
valid['image_id'] = valid['image_id'].str.replace('.jpg','')
valid = valid[~valid['image_id'].str.contains('[^0-9]')]  # Zostaw tylko wartości zawierające cyfry
valid['image_id'] = valid['image_id'].astype(int)


#połączenie danych
train_data = pd.merge(train,df,on='image_id',how='outer')
train_data = train_data[['filename','category_id']]
train_data.columns = ['filename','label']

valid_data = pd.merge(valid,df_valid,on='image_id',how='outer')
valid_data = valid_data[['filename','category_id']]
valid_data.columns = ['filename','label']

train_data['filename'] = train_data['filename'].str.replace(project_path + 'train/','')
valid_data['filename'] = valid_data['filename'].str.replace(project_path + 'valid/','')

train_data['label'] = train_data['label'].replace({1:0,2:1})
valid_data['label'] = valid_data['label'].replace({1:0,2:1})


# transformacje obrazków
train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


storage_client = storage.Client()

bucket_name = 'wtum_files' 
bucket = storage_client.bucket(bucket_name)

images = []
labels = []

# Iteracja przez DataFrame
for i, annotation in train_data.iterrows():
    image_name = annotation['filename']
    target = annotation['label']
    blob = bucket.blob(image_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = train_transforms(image)
    images.append(image)
    labels.append(torch.tensor(target, dtype=torch.long))

# Tworzenie tensorów z list obrazów i etykiet
image_tensor = torch.stack(images)
target_tensor = torch.stack(labels)


# Stworzenie TensorDataset
train_dataset = torch.utils.data.TensorDataset(image_tensor, target_tensor)


images = []
labels = []

# Iteracja przez DataFrame
for i, annotation in valid_data.iterrows():
    image_name = annotation['filename']
    target = annotation['label']
    blob = bucket.blob(image_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = valid_transforms(image)
    images.append(image)
    labels.append(torch.tensor(target, dtype=torch.long))

# Tworzenie tensorów z list obrazów i etykiet
image_tensor = torch.stack(images)
target_tensor = torch.stack(labels)


# Stworzenie TensorDataset
valid_dataset = torch.utils.data.TensorDataset(image_tensor, target_tensor)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)




#pętla treningowa
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    for epoch in range(n_epochs):
        model.train()
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        epoch_loss = loss_train / len(train_loader)

        if (epoch + 1) % 1 == 0:
            #walidacja modelu
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():

                for imgs, labels in val_loader:
                  imgs = imgs.to(device)
                  labels = labels.to(device)

                  outputs = model(imgs)
                  loss = loss_fn(outputs, labels)
                  val_loss += loss.item()
                  preds = torch.argmax(outputs, dim=1)
                  total += labels.shape[0]
                  correct += int((preds == labels).sum())

                val_loss /= len(val_loader.dataset)
                val_accuracy = correct / total

            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')





# walidacja dokładności modelu
def validate(model, train_loader, val_loader):
    model.eval()
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                total += labels.shape[0]
                correct += int((preds == labels).sum())

        print(f"{name} accuracy: {correct/total}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_resnet18 = models.resnet18(pretrained=True)
num_features = model_resnet18.fc.in_features
model_resnet18.fc = torch.nn.Linear(num_features, 2)

model_resnet18.to(device)



# trening resnet18
optimizer = torch.optim.SGD(model_resnet18.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()



training_loop(
    n_epochs = 10,
    optimizer = optimizer,
    model = model_resnet18,
    loss_fn = loss_fn,
    train_loader = train_loader,
    val_loader = valid_loader
)



validate(model_resnet18, train_loader, valid_loader)



with tempfile.NamedTemporaryFile() as tmp:
    torch.save({
        'model_state_dict': model_resnet18.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, tmp.name)

    # Inicjalizacja klienta Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket('wtum_files')
    blob = bucket.blob('model_resnet18.pth')

    # Przesłanie pliku do bucketu GCS
    blob.upload_from_filename(tmp.name)
