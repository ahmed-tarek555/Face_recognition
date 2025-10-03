import os
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'RGB'
mtcnn = MTCNN(image_size=128)
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, targets):
    batch = torch.randint(data.shape[0], (batch_size, ))
    x = data[batch]
    y = targets[batch]
    return x, y

def process_img(path):
    if isinstance(path, Image.Image):
        img = path.convert(target_format)
    else:
        img = Image.open(path).convert(target_format)
    img = mtcnn(img)
    if img is not None:
        print(f'Image loaded of shape {img.shape}')
    return img

def get_embedding_probs(pic, model):
    img = process_img(pic)
    if img is None:
        return None, None
    img = torch.stack((img,), dim=0)
    img_embedding, probs = model(img)
    img_embedding = img_embedding.squeeze(0)
    return F.normalize(img_embedding, dim=0), probs

with torch.no_grad():
    def eval_loss(model, path):
        distances = []
        embeddings1 = get_eval_embeddings(model, path, 0)
        embeddings2 = get_eval_embeddings(model, path, 1)
        embeddings = list(zip(embeddings1.values(), embeddings2.values()))
        for emb1, emb2 in embeddings:
            distances.append(torch.norm(emb1 - emb2))
        loss = sum(distances) / len(distances)
        return loss

def get_eval_embeddings(model, path, pic_index):
    embeddings = {}

    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filename = os.listdir(person_dir)[pic_index]
        pic = os.path.join(person_dir, filename)
        img_embedding, probs = get_embedding_probs(pic, model)
        if img_embedding is None:
            continue
        else:
            embeddings[person_name] = img_embedding
    return embeddings
