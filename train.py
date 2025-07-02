import torch
import os
from facenet_pytorch import MTCNN
from model import Model

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'RGB'
dataset_path = 'data_set'
mtcnn = MTCNN(image_size=128)

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

            img = torch.load(file_path, weights_only=False)

            x.append(img)
            y.append(current_label)
        current_label += 1
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y, labels

data, targets, labels = load_dataset('processed_train_dataset')

m = Model(len(labels))

m._train(data, targets, 10000,  1e-3)

path = 'parameters.pth'

torch.save(m.state_dict(), path)
