import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import shutil

from recognize import recognize
from store_embeddings import store_embeddings


class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        self.recognize_button = Button(master, text="Recognize", command=self.recognize)
        self.recognize_button.pack(pady=20)

        self.add_user_button = Button(master, text="Add user", command=self.add_user)
        self.add_user_button.pack(pady=10)

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.register_users_button = Button(master, text="Register users", command=self.register_users)
        self.register_users_button.pack(pady=30)

        self.result_label = Label(master, text="")
        self.result_label.pack()


    def add_user(self):
        save_dir = 'test/register'
        name = self.entry.get()
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
        self.result_label.config(text=f"{name}")


root = tk.Tk()
root.geometry("800x600")
app = FaceApp(root)
root.mainloop()
