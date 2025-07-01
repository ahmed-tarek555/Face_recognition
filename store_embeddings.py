import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from model import Model
from facenet_pytorch import MTCNN

target_format = 'RGB'
mtcnn = MTCNN(image_size=128)
m = Model(105)

def process_img(path):
    img = Image.open(path).convert(target_format)
    img = mtcnn(img)
    if img is not None:
        print(f'Image loaded of shape {img.shape}')

    return img

def get_embedding(pic):
    img = process_img(pic)
    if img is None:
        return None
    img = torch.stack((img,), dim=0)
    img_embedding = m(img)
    img_embedding = img_embedding.squeeze(0)
    return F.normalize(img_embedding, dim=0)

def store_embeddings(path, pic_index, save_dir="known_embeddings"):
    m.eval()
    os.makedirs(save_dir, exist_ok=True)

    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filenames = sorted(os.listdir(person_dir))
        filename = filenames[pic_index]
        pic = os.path.join(person_dir, filename)

        img_embedding = get_embedding(pic)
        if img_embedding is None:
            continue

        person_embedding_dir = os.path.join(save_dir, person_name)
        os.makedirs(person_embedding_dir, exist_ok=True)

        embedding_file = os.path.join(person_embedding_dir, f"{person_name}.pt")
        torch.save(img_embedding, embedding_file)
        print('Done')

store_embeddings('test/register', 0)