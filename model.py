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
mtcnn = MTCNN(image_size=128)

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

class Model(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv_block(3, 32, 3, 1),
                                  conv_block(32, 64, 3, 2),
                                  conv_block(64, 128, 3, 2),
                                  conv_block(128, 256, 3, 2),
                                  conv_block(256, 512, 3, 2),
                                  )

        self.projection = nn.Sequential(nn.Conv2d(3, 512, 1, 1),
                                        nn.AdaptiveAvgPool2d((6, 6)),
                                        )

        self.fc = nn.Sequential(nn.Linear(512*6*6, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden // 2),
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

    def _train(self, data, targets, n_iter, lr):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr)
        current_iter = 0
        for i in range(n_iter):
            batch = torch.randint(0, data.shape[0], (batch_size,))
            x = data[batch]
            y = targets[batch]

            # forward pass
            loss, embeddings = self(x, y)

            # backward pass
            optim.zero_grad()
            loss.backward()
            # print(self.fc[0].weight.grad)

            # update
            optim.step()

            print(loss)
            current_iter += 1
            print(int((current_iter/n_iter)*100))
        self.eval()

if __name__ == "__main__":
    m = Model(105)
    m.load_state_dict(torch.load('parameters.pth'))

def process_img(path):
    img = Image.open(path).convert(target_format)
    img = mtcnn(img)
    if img is not None:
        print(f'Image loaded of shape {img.shape}')
    return img

with torch.no_grad():
    def eval_loss(path):
        m.eval()
        distances = []
        embeddings1 = store_embeddings(path, 0)
        embeddings2 = store_embeddings(path, 1)
        embeddings = list(zip(embeddings1.values(), embeddings2.values()))
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

def main():
    print(f'Number of parameters is: {sum(p.numel() for p in m.parameters())}')
    print(f'evaluation loss is {eval_loss("eval")}')

if __name__ == "__main__":
    main()

