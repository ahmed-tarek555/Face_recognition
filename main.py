import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

batch_size = 200
n_hidden = 200
target_size = (64, 64)
target_format = 'L'
dataset_path = 'data_set'

def process_img(path):
    img = Image.open(path).convert(target_format)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img

def load_dataset(dataset_path, target_size):
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

            img = process_img(file_path)

            x.append(img)
            y.append(current_label)

        current_label += 1

    x = torch.tensor(np.array(x), dtype=torch.float32)
    y = torch.tensor(y)
    return x, y, labels

data, targets, labels = load_dataset('data_set', target_size)

data = data.view(data.shape[0], -1)

class Model:
    def __init__(self):
        self.w1 = torch.randn(64*64, n_hidden) * 0.1
        self.b1 = torch.randn(n_hidden) * 0
        self.w2 = torch.randn(n_hidden, n_hidden//2) * 0.1
        self.b2 = torch.randn(n_hidden//2) * 0
        self.w3 = torch.randn(n_hidden//2, len(labels))

        parameters = [self.w1] + [self.w2] + [self.w3] + [self.b1] + [self.b2]
        for p in parameters:
            p.requires_grad = True


    def __call__(self, x, y=None):
        x = torch.tanh(x @ self.w1 + self.b1)
        x = torch.tanh(x @ self.w2 + self.b2)
        embeddings = x
        logits = x @ self.w3

        if y is not None:
            # loss = -probs[range(0, probs.shape[0]), y].log().mean()
            loss = F.cross_entropy(logits, y)
            return loss, embeddings
        else:
            return embeddings

    def parameters(self):
        return [self.w1]+[self.b1]+[self.w2]+[self.b2]+[self.w3]

m = Model()

# lr = 0.01
# n_iter = 1000
# for i in range(n_iter):
#
#     batch = torch.randint(0, data.shape[0], (batch_size,))
#     x = data[batch]
#     y = targets[batch]
#
#     loss, embeddings = m(x, y)
#
#     # backward pass
#     for p in m.parameters():
#         p.grad = None
#     loss.backward()
#
#     # update
#     for p in m.parameters():
#         p.data += -lr * p.grad
#
#     print(loss)

def get_embedding(pic):
    img = process_img(pic)
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

known_embeddings = store_embeddings('data_set')


def recognize(pic):
    best_match = None
    best_distance = float('inf')

    img_embedding = get_embedding(pic)
    for name, embedding in known_embeddings.items():
        distance = torch.norm(img_embedding, embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = name

    return best_match











