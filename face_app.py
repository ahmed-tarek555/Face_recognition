import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import shutil
import cv2
from PIL import Image

from recognize import recognize
from store_embeddings import store_embeddings


class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        self.camera_recognize_button = Button(master, text='Camera recognition', command=self.camera_detection)
        self.camera_recognize_button.pack()

        self.camera_source_entry = tk.Entry(root)
        self.camera_source_entry.pack()

        self.recognize_button = Button(master, text="Recognize", command=self.recognize)
        self.recognize_button.pack(pady=20)

        self.add_user_button = Button(master, text="Add user", command=self.add_user)
        self.add_user_button.pack(pady=10)

        self.pic_entry = tk.Entry(root)
        self.pic_entry.pack()

        self.register_users_button = Button(master, text="Register users", command=self.register_users)
        self.register_users_button.pack(pady=30)

        self.result_label = Label(master, text="")
        self.result_label.pack()


    def add_user(self):
        save_dir = 'test/register'
        name = self.pic_entry.get()
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
        name = recognize(file_path)
        if name is None:
            self.result_label.config(text="Couldn't detect a face")
        else:
            self.result_label.config(text=f"{name}")

    def camera_detection(self):
        global name
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
                name = recognize(pic)

            cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Video Face Recognition", frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


root = tk.Tk()
root.geometry("800x600")
app = FaceApp(root)
root.mainloop()
