import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torch.nn.functional import embedding

batch_size = 200
n_hidden = 200
target_size = (64, 64)
target_format = 'L'
dataset_path = 'data_set'

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

            img = Image.open(file_path).convert(target_format)
            img = img.resize(target_size)
            img = np.array(img)/255.0

            x.append(img)
            y.append(current_label)

        current_label += 1

    x = torch.tensor(np.array(x), dtype=torch.float32)
    y = torch.tensor(y)
    return x, y, labels

X, Y, labels = load_dataset(dataset_path, target_size)



X = X.view(X.shape[0], -1)

# layer 1
w1 = torch.randn(64*64, n_hidden) * 0.1
b1 = torch.randn(n_hidden) * 0

# layer 2
w2 = torch.randn(n_hidden, n_hidden//2) * 0.1
b2 = torch.randn(n_hidden//2) * 0

# layer 3
w3 = torch.randn(n_hidden//2, len(labels))

parameters = [w1]+[b1]+[w2]+[b2]+[w3]
for p in parameters:
    p.requires_grad = True

def forward_pass(x, y=None):

    x = x @ w1 + b1
    x = x.tanh()
    x = x @ w2 + b2
    x = x.tanh()
    embeddings = x
    logits = x @ w3
    probs = torch.softmax(logits, dim=1)

    if y is not None:
        # loss = -probs[range(0, probs.shape[0]), y].log().mean()
        loss = F.cross_entropy(logits, y)
        return loss, embeddings
    else:
        return embeddings

# lr = 0.01
# n_iter = 10000
# for i in range(n_iter):
#
#     batch = torch.randint(0, X.shape[0], (batch_size,))
#     x = X[batch]
#     y = Y[batch]
#
#     loss, embeddings = forward_pass(x, y)
#
#     # backward pass
#     for p in parameters:
#         p.grad = None
#     loss.backward()
#
#     # update
#     for p in parameters:
#         p.data += -lr * p.grad
#
#     print(loss)


def get_embeddings(dataset_path):
    embeddings = {}

    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filename = os.listdir(person_dir)[0]
        file_path = os.path.join(person_dir, filename)
        img = Image.open(file_path).convert(target_format)
        img = img.resize(target_size)
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = img.view(-1)

        img = torch.stack((img,), dim=0)
        img_embedding = forward_pass(img)
        embeddings[person_name] = img_embedding.squeeze(0)

    return embeddings

get_embeddings(dataset_path)












