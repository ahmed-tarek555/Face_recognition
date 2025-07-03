import tkinter as tk
from tkinter import filedialog, Label, Button
from recognize import recognize


class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        self.label = Label(master, text="Upload an image:")
        self.label.pack()

        self.upload_button = Button(master, text="Browse", command=self.upload_image)
        self.upload_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

        self.image_label = Label(master)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Display image
        # img = Image.open(file_path).convert("RGB")
        # img_resized = img.resize((200, 200))
        # photo = ImageTk.PhotoImage(img_resized)
        # self.image_label.config(image=photo)
        # self.image_label.image = photo

        name = recognize(file_path)

        self.result_label.config(text=f"This person is {name}")

root = tk.Tk()
app = FaceApp(root)
root.mainloop()
