import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from PIL import Image

batch_size = 20
n_hidden = 100
target_size = (128, 128)
target_format = 'L'
dataset_path = 'data_set'

def process_img(path, target_size):
    img = Image.open(path).convert(target_format)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img

def load_dataset(dataset_path):
    x = []
    y = []
    labels = []
    current_label = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        labels.append(person_name)

        for filename in os.listdir(person_dir):
            file_path = os.path.join(person_dir, filename)

            img = process_img(file_path, target_size)

            x.append(img)
            y.append(current_label)
        current_label += 1

    x = torch.tensor(np.array(x), dtype=torch.float32)
    y = torch.tensor(y)
    return x, y, labels

data, targets, labels = load_dataset('data_set')

data = data.view(data.shape[0], -1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(128*128, n_hidden),
                                nn.Tanh(),
                                nn.Linear(n_hidden, n_hidden//2),
                                nn.BatchNorm1d(n_hidden//2),
                                nn.Tanh(),
                                )
        self.classifier = nn.Linear(n_hidden//2, len(labels), bias=False)

    def forward(self, x, y=None):
        x = self.fc(x)
        embeddings = x
        logits = self.classifier(x)

        if y is not None:
            # loss = -probs[range(0, probs.shape[0]), y].log().mean()
            loss = F.cross_entropy(logits, y)
            return loss, embeddings
        else:
            return embeddings

    def _train(self, n_iter, lr):
        self.train()
        optim = torch.optim.AdamW(m.parameters(), lr)
        for i in range(n_iter):
            batch = torch.randint(0, data.shape[0], (batch_size,))
            x = data[batch]
            y = targets[batch]

            # forward pass
            loss, embeddings = m(x, y)

            # backward pass
            optim.zero_grad()
            loss.backward()

            # update
            optim.step()

            print(loss)
        self.eval()

m = Model()
m._train(1000, 0.01)

def get_embedding(pic):
    img = process_img(pic, target_size)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.view(-1)
    img = torch.stack((img,), dim=0)
    img_embedding = m(img)
    img_embedding = img_embedding.squeeze(0)
    return img_embedding


def store_embeddings(path):
    embeddings = {}

    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filename = os.listdir(person_dir)[0]
        pic = os.path.join(person_dir, filename)
        embeddings[person_name] = get_embedding(pic)

    return embeddings

known_embeddings = store_embeddings('test/register')


def recognize(pic):
    best_match = None
    best_distance = float('inf')

    img_embedding = get_embedding(pic)
    for name, embedding in known_embeddings.items():
        distance = torch.norm(img_embedding - embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = name

    return best_match

print(recognize('test/test_faces/Ahmed.jpg'))

