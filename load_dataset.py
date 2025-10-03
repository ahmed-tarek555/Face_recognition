import torch
import os
from proj_utils import process_img


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

preprocess_and_save('data/gender_dataset', 'processed_data/processed_gender_dataset')