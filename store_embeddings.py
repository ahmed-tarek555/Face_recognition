import os
import torch
import torch.nn.functional as F
from model import Model
from utils import get_embedding_probs
from facenet_pytorch import MTCNN

target_format = 'RGB'
mtcnn = MTCNN(image_size=128)
m = Model(105)
m.load_state_dict(torch.load('identity_gender_model.pth'))


def store_embeddings(path, save_dir="known_embeddings"):
    m.eval()
    os.makedirs(save_dir, exist_ok=True)

    for person_name in sorted(os.listdir(path)):
        embeddings = []
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue

        filenames = sorted(os.listdir(person_dir))
        for filename in filenames:
            pic = os.path.join(person_dir, filename)
            img_embedding, probs = get_embedding_probs(pic, m)
            if img_embedding is None:
                continue
            embeddings.append(img_embedding)
        ave_embedding = torch.stack(embeddings, dim=0).mean(dim=0, keepdim=False)
        ave_embedding = F.normalize(ave_embedding, dim=0)
        person_embedding_dir = os.path.join(save_dir, person_name)
        os.makedirs(person_embedding_dir, exist_ok=True)
        embedding_file = os.path.join(person_embedding_dir, f"{person_name}.pt")
        torch.save(ave_embedding, embedding_file)
        print('Done')

# store_embeddings('test/register')