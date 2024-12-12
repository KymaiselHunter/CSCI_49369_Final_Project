import cv2
import numpy as np
from collections import Counter
import math
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

#Future add, when it is detected a key is pressed in the image, have that output in the keyboard image
#Also, based on keyboard size (white keys and black keys), we will hopefully create a virtual keyboard just on that

#def keyboardCreation(white_keys, black_keys)

KEY_COORDINATES = {
    "C": 10,   # X-coordinate for "C"
    "C#": 85,  # X-coordinate for "C#"
    "D": 50,   # X-coordinate for "D"
    "D#": 170,  # X-coordinate for "D#"
    "E": 90,   # X-coordinate for "E"
    "F": 130,  # X-coordinate for "F"
    "F#": 340, # X-coordinate for "F#"
    "G": 170,  # X-coordinate for "G"
    "G#": 430, # X-coordinate for "G#"
    "A": 210,  # X-coordinate for "A"
    "A#": 510, # X-coordinate for "A#"
    "B": 250,  # X-coordinate for "B"
}


class virtualKeyboard:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.geometry('600x400')
        self.window.title('Virtual Keyboard')

        self.keyboard_image = Image.open('./KeyImages/smallKey.png').resize((600, 400))
        self.keyboard_tk = ImageTk.PhotoImage(self.keyboard_image)

        # Canvas (so we can add things on top of the current keyboard)
        self.canvas = Canvas(self.window, width=600, height=400)
        self.canvas.pack()

        # Add keyboard image
        self.canvas.create_image(0, 0, image=self.keyboard_tk, anchor="nw")

        self.notes_widgets = {}

        self.green_key_image = Image.open('./KeyImages/key_green_top.png').resize((50, 400))  # Adjust size
        self.green_key_tk = ImageTk.PhotoImage(self.green_key_image)

    def start(self):
        """Start the Tkinter main loop."""
        self.window.mainloop()

    def add_note(self, key: str, is_green: bool = True) -> None:
        x = KEY_COORDINATES[key]
        y = 200  

        
        key_image = self.green_key_tk 

        
        self.canvas.create_image(x, y, image=key_image, anchor="center")








