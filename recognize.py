import torch
import torch.nn.functional as F
import os
from PIL import Image
from facenet_pytorch import MTCNN
from model import Model

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'RGB'
dataset_path = 'data_set'
mtcnn = MTCNN(image_size=128)
m = Model(105)


def load_known_embeddings(path):
    known_embeddings = {}
    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue
        files = sorted(os.listdir(person_dir))
        file = files[0]
        file_path = os.path.join(person_dir, file)
        embedding = torch.load(file_path, weights_only=False)
        known_embeddings[person_name] = embedding
    return known_embeddings

known_embeddings = load_known_embeddings('known_embeddings')

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

def recognize(pic):
    m.eval()
    best_match = None
    best_distance = float('inf')
    threshold = 0.6

    img_embedding = get_embedding(pic)
    if img_embedding is None:
        return "Couldn't detect a face"
    for name, embedding in known_embeddings.items():
        distance = torch.norm(img_embedding - embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = name
    if best_distance > threshold:
        best_match = 'Unknown'
    return best_match

img = os.listdir('test/test_faces/being_tested')[0]
best_match = recognize(f'test/test_faces/being_tested/{img}')
print(f'This person is {best_match}')