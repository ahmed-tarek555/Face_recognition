import torch
import os
import numpy as np
from PIL import Image


target_size = (64, 64)
target_format = 'L'


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

x, y, labels = load_dataset('data_set', target_size)

n_hidden = 200

x = x.view(x.shape[0], -1)

# layer 1
w1 = torch.randn(64*64, n_hidden)
b1 = torch.randn(n_hidden)

# layer 2
w2 = torch.randn(n_hidden, n_hidden//2)
b2 = torch.randn(n_hidden//2)

# layer 3
w3 = torch.randn(n_hidden//2, len(labels))

parameters = [w1]+[b1]+[w2]+[b2]+[w3]
for p in parameters:
    p.requires_grad = True

# forward pass
x = x @ w1 + b1
x = x @ w2 + b2
logits = x @ w3

probs = logits / torch.sum(logits, 1, keepdim=True)

loss = -probs[range(0, probs.shape[0]), y].log().mean()

# backward pass
for p in parameters:
    p.grad = None
loss.backward()

# update
lr = 0.1
for p in parameters:
    p.data -= lr * p.grad


print(loss)

