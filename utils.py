import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

batch_size = 32
n_hidden = 200
target_size = (128, 128)
target_format = 'RGB'
mtcnn = MTCNN(image_size=128)


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
