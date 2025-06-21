import torch
import os
import numpy as np
from PIL import Image


target_size = (128, 128)
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

print(y)
