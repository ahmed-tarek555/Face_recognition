import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import shutil
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from recognize import recognize
from store_embeddings import store_embeddings
from utils import get_embedding_probs
from model import Model

m = Model(105)
m.load_state_dict(torch.load('identity_gender_model.pth'))
mtcnn = MTCNN(image_size=128)


class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        self.cameras_label = Label(text='Camera registration/ recognition')
        self.cameras_label.grid(row=0, column=0, padx=50, pady=50)

        self.camera_source_label = Label(text='Enter the camera source')
        self.camera_source_label.grid(row=1, column=0, padx=50, pady=0)

        self.camera_source_entry = tk.Entry(root)
        self.camera_source_entry.grid(row=2, column=0, padx=50, pady=0)

        self.camera_register_label = Label(text='Enter the name of the user')
        self.camera_register_label.grid(row=3, column=0, padx=50, pady=(30,0))

        self.camera_register_entry = tk.Entry(root)
        self.camera_register_entry.grid(row=4, column=0, padx=50, pady=(0,10))

        self.camera_register_button = Button(master, text='Camera register', command=self.camera_register)
        self.camera_register_button.grid(row=5, column=0, padx=50, pady=(0,10))

        self.camera_recognize_button = Button(master, text='Camera recognition', command=self.camera_detection)
        self.camera_recognize_button.grid(row=6, column=0, padx=50, pady=20)

        self.manual_label = Label(text='Manual registration/ recognition')
        self.manual_label.grid(row=0, column=30, padx=50, pady=50)

        self.pic_entry_label = Label(text='Enter the name of the user')
        self.pic_entry_label.grid(row=1, column=30, padx=50, pady=0)

        self.pic_entry = tk.Entry(root)
        self.pic_entry.grid(row=2, column=30, padx=50, pady=0)

        self.add_user_button = Button(master, text="Add user", command=self.add_user)
        self.add_user_button.grid(row=3, column=30, padx=50, pady=(0, 30))

        self.register_users_button = Button(master, text="Register users", command=self.register_users)
        self.register_users_button.grid(row=4, column=30, padx=50, pady=(0, 30))

        self.recognize_button = Button(master, text="Recognize", command=self.recognize)
        self.recognize_button.grid(row=5, column=30, padx=50, pady=0)

        self.result_label = Label(master, text="")
        self.result_label.grid(row=6, column=30, padx=50, pady=30)


    def add_user(self):
        save_dir = 'test/register'
        name = self.pic_entry.get()
        if name == '':
            self.pic_entry_label.config(text='Please enter the name first')
            return
        pic_dir = filedialog.askopenfilename()
        save_pic_dir = os.path.join(save_dir, name)
        os.makedirs(save_pic_dir, exist_ok=True)
        shutil.copy(pic_dir, save_pic_dir)

    def register_users(self):
        store_embeddings('test/register')
        self.register_users_button.config(text='Done')

    def recognize(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        name, gender = recognize(file_path)
        if name is None:
            self.result_label.config(text="Couldn't detect a face")
        else:
            self.result_label.config(text=f"{name}-{gender}")

    def camera_detection(self):
        global name, gender
        source = self.camera_source_entry.get()
        cap = cv2.VideoCapture(f"{source}")

        frame_count = 0
        process_every_n = 15
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pic = Image.fromarray(rgb)
                name, gender = recognize(pic)

            cv2.putText(frame, f'{name}-{gender}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Video Face Recognition", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def camera_register(self):
        m.eval()
        name = self.camera_register_entry.get()
        if name == '':
            self.camera_register_label.config(text='Please enter the name first')
            return
        source = self.camera_source_entry.get()
        cap = cv2.VideoCapture(f"{source}")
        frame_count = 0
        process_every_n = 15
        embeddings = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pic = Image.fromarray(rgb)

                if mtcnn(pic) is not None:
                    embedding, probs = get_embedding_probs(pic, m)
                    embeddings.append(embedding)


            if len(embeddings) == 5:
                ave_embedding = torch.stack(embeddings, dim=0).mean(dim=0, keepdim=False)
                ave_embedding = F.normalize(ave_embedding, dim=0)
                person_embedding_dir = f'known_embeddings/{name}'
                os.makedirs(person_embedding_dir, exist_ok=True)
                embedding_file = os.path.join(person_embedding_dir, f"{name}.pt")
                torch.save(ave_embedding, embedding_file)
                print('Done')
                break

            cv2.imshow("Video Face Recognition", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

root = tk.Tk()
root.geometry("600x500")
app = FaceApp(root)
root.mainloop()
