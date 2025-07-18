import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from facenet_pytorch import MTCNN
from utils import get_embedding_probs

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
        self.identity = nn.Linear(n_hidden // 2, n_out, bias=False)

        self.gender = nn.Linear(n_hidden//2, 2)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data *= 0.1

    def forward(self, x, id=None, gender=None):
        iden = x
        x = self.conv(x)
        iden = self.projection(iden)
        x = x + iden
        A, B, C, D = x.shape
        x = x.reshape(A, -1)
        x = self.fc(x)
        embeddings = x
        id_logits = self.identity(x)
        gender_logits = self.gender(x)
        probs = F.softmax(gender_logits, 1)

        if id is not None:
            # loss = -probs[range(0, probs.shape[0]), y].log().mean()
            loss = F.cross_entropy(id_logits, id)
            return loss
        if gender is not None:
            loss = F.cross_entropy(gender_logits, gender)
            return loss

        return embeddings, probs


    def _train(self, n_iter, lr, iden_data=None, iden_targets=None, gender_data=None, gender_targets=None):
        global loss
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr)
        current_iter = 0
        for i in range(n_iter):
            if i % 2 == 0:
                batch = torch.randint(0, iden_data.shape[0], (batch_size,))
                x = iden_data[batch]
                y = iden_targets[batch]
                loss = self(x, id=y, gender=None)
                print(f'iden loss is:{loss}')

            elif i % 2 != 0:
                batch = torch.randint(0, gender_data.shape[0], (batch_size,))
                x = gender_data[batch]
                y = gender_targets[batch]
                loss = self(x,id=None, gender=y)
                print(f'gender loss is:{loss}')

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
    m.load_state_dict(torch.load('models/identity_gender_model.pth'))


with torch.no_grad():
    def eval_loss(path):
        m.eval()
        distances = []
        embeddings1 = get_eval_embeddings(path, 0)
        embeddings2 = get_eval_embeddings(path, 1)
        embeddings = list(zip(embeddings1.values(), embeddings2.values()))
        for emb1, emb2 in embeddings:
            distances.append(torch.norm(emb1 - emb2))
        loss = sum(distances) / len(distances)
        return loss


def get_eval_embeddings(path, pic_index):
    m.eval()
    embeddings = {}

    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filename = os.listdir(person_dir)[pic_index]
        pic = os.path.join(person_dir, filename)
        img_embedding, probs = get_embedding_probs(pic, m)
        if img_embedding is None:
            continue
        else:
            embeddings[person_name] = img_embedding
    return embeddings

def main():
    print(f'Number of parameters is: {sum(p.numel() for p in m.parameters())}')
    print(f'evaluation loss is {eval_loss("data/eval")}')

if __name__ == "__main__":
    main()

