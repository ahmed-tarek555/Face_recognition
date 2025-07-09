import torch
from facenet_pytorch import MTCNN
import os
from utils import process_img

target_format = 'RGB'
mtcnn = MTCNN(image_size=128)

def preprocess_and_save(raw_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for person in os.listdir(raw_dir):
        person_dir = os.path.join(raw_dir, person)
        save_person_dir = os.path.join(save_dir, person)
        os.makedirs(save_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            face = process_img(img_path)

            if face is not None:
                save_path = os.path.join(save_person_dir, img_name.split('.')[0] + ".pt")
                torch.save(face, save_path)
    print('Done')

preprocess_and_save('gender_dataset', 'processed_gender_dataset')