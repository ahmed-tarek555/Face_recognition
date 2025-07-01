import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import Image
from facenet_pytorch import MTCNN

# CONVOLUTIONAL LAYER FORMULA: [(in−K+2P)/S]+1

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'RGB'
dataset_path = 'data_set'
mtcnn = MTCNN(image_size=128)

class Model(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 5, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(5, 3, 2, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(3, 1, 3, 1),
                                  nn.ReLU())

        self.projection = nn.Conv2d(3, 1, 15, 4, 1)

        self.fc = nn.Sequential(nn.Linear(841, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden // 2),
                                nn.LayerNorm(n_hidden // 2),
                                nn.ReLU(),
                                nn.Linear(n_hidden // 2, n_hidden // 2),
                                nn.LayerNorm(n_hidden // 2),
                                )
        self.classifier = nn.Linear(n_hidden // 2, n_out, bias=False)

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
            loss, embeddings = self(x, y)

            # backward pass
            optim.zero_grad()
            loss.backward()

            # update
            optim.step()

            print(loss)
        self.eval()

def process_img(path):
    img = Image.open(path).convert(target_format)
    img = mtcnn(img)
    if img is not None:
        print(f'Image loaded of shape {img.shape}')

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

            img = torch.load(file_path, weights_only=False)

            x.append(img)
            y.append(current_label)
        current_label += 1
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y, labels

with torch.no_grad():
    def eval_loss(path):
        m.eval()
        embeddings1 = store_embeddings(path, 0)
        embeddings2 = store_embeddings(path, 1)
        embeddings = list(zip(embeddings1.values(), embeddings2.values()))
        distances = []
        for emb1, emb2 in embeddings:
            distances.append(torch.norm(emb1 - emb2))
        loss = sum(distances) / len(distances)
        return loss

def get_embedding(pic):
    img = process_img(pic)
    if img is None:
        return None
    img = torch.stack((img,), dim=0)
    img_embedding = m(img)
    img_embedding = img_embedding.squeeze(0)
    return F.normalize(img_embedding, dim=0)


def store_embeddings(path, pic_index):
    m.eval()
    embeddings = {}

    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filename = os.listdir(person_dir)[pic_index]
        pic = os.path.join(person_dir, filename)
        img_embedding = get_embedding(pic)
        if img_embedding is None:
            continue
        else:
            embeddings[person_name] = img_embedding
    return embeddings

# def recognize(pic):
#     m.eval()
#     best_match = None
#     best_distance = float('inf')
#     threshold = 0.6
#
#     img_embedding = get_embedding(pic)
#     if img_embedding is None:
#         return "Couldn't detect a face"
#     for name, embedding in known_embeddings.items():
#         distance = torch.norm(img_embedding - embedding)
#         if distance < best_distance:
#             best_distance = distance
#             best_match = name
#     if best_distance > threshold:
#         best_match = 'Unknown'
#     return best_match

def main():
    data, targets, labels = load_dataset('processed_train_dataset')
    m = Model(len(labels))
    print(sum(p.numel() for p in m.parameters()))
    m._train(10000, 1e-3)
    print(f'eval loss is {eval_loss("eval")}')

if __name__ == "__main__":
    main()

