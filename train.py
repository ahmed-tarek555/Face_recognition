import torch
import os
from model import Model, parameters_file
from proj_utils import get_batch, device

lr = 3e-4
n_iter = 10000

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

id_data, id_targets, id_labels = load_dataset('processed_data/processed_identity_dataset')
gender_data, gender_targets, gender_labels = load_dataset('processed_data/processed_gender_dataset')

m = Model(len(id_labels))
m = m.to(device)

global loss
m.train()
optim = torch.optim.AdamW(m.parameters(), lr)
current_iter = 0
for i in range(n_iter):
    if i % 2 == 0:
        x, y = get_batch(id_data, id_targets)
        x = x.to(device)
        y = y.to(device)
        loss = m(x, id=y, gender=None)
        print(f'iden loss is:{loss}')

    elif i % 2 != 0:
        x, y = get_batch(gender_data, gender_targets)
        x = x.to(device)
        y = y.to(device)
        loss = m(x, id=None, gender=y)
        print(f'gender loss is:{loss}')

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)
    current_iter += 1
    print(int((current_iter / n_iter) * 100))

torch.save(m.state_dict(), parameters_file)
