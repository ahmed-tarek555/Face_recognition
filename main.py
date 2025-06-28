import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from PIL import Image

# CONVOLUTIONAL LAYER FORMULA: [(in−K+2P)/S]+1

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'L'
dataset_path = 'data_set'

def process_img(path, target_size):
    img = Image.open(path).convert(target_format)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    if target_format == 'L':
        img = torch.stack((img,), dim=2)
    img = img.permute(2, 0, 1).contiguous()
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 5, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(5, 3, 2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(3, 1, 3, 1),
                                  nn.ReLU())

        self.projection = nn.Conv2d(1, 1, 15, 4, 1)

        self.fc = nn.Sequential(nn.Linear(841, n_hidden),
                                nn.ReLU(),
                                # nn.Linear(n_hidden, n_hidden//2),
                                # nn.LayerNorm(n_hidden//2),
                                # nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden // 2),
                                nn.LayerNorm(n_hidden // 2),
                                nn.ReLU()
                                )
        self.classifier = nn.Linear(n_hidden // 2, len(labels), bias=False)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data *= 0.1

    def forward(self, x, y=None):
        iden = x
        x = self.conv(x)
        iden = self.projection(iden)
        x = x + iden
        A, B, C, D = x.shape
        x = x.reshape(A, -1)
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
print(sum(p.numel() for p in m.parameters()))
m._train(10000, 1e-3)

def get_embedding(pic):
    img = process_img(pic, target_size)
    img = torch.stack((img,), dim=0)
    img_embedding = m(img)
    img_embedding = img_embedding.squeeze(0)
    return img_embedding


def store_embeddings(path):
    m.eval()
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
    m.eval()
    best_match = None
    best_distance = float('inf')
    threshold = 0.5

    img_embedding = get_embedding(pic)
    for name, embedding in known_embeddings.items():
        distance = torch.norm(img_embedding - embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = name
    if best_distance > threshold:
        best_match = 'Unknown'
    return best_match

print(recognize('test/test_faces/Hatshepsut.jpg'))
