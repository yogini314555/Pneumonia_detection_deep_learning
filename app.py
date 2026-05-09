import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2

# =========================
# BUILD MODEL
# =========================
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(96,96,3)
)

base_model.trainable = False

model = Sequential()

model.add(base_model)

model.add(Flatten())

# MUST MATCH TRAINED MODEL
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))




# =========================
# LOAD WEIGHTS
# =========================

model.load_weights("best_model.weights.h5")

# =========================
# CREATE WINDOW
# =========================

root = tk.Tk()

root.title("Pneumonia Detection System")

root.geometry("750x700")

root.configure(bg="white")

# =========================
# HEADING
# =========================

heading = tk.Label(
    root,
    text="Pneumonia Detection Using Deep Learning",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="darkblue"
)

heading.pack(pady=20)

# =========================
# IMAGE LABEL
# =========================

image_label = tk.Label(root, bg="white")

image_label.pack(pady=20)

# =========================
# RESULT LABEL
# =========================

result_label = tk.Label(
    root,
    text="Upload Chest X-Ray Image",
    font=("Arial", 18),
    bg="white",
    fg="green"
)

result_label.pack(pady=20)

# =========================
# PREDICTION FUNCTION
# =========================

def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if file_path:

        try:

            # Read image
            img = cv2.imread(file_path)

            if img is None:

                result_label.config(
                    text="Invalid Image",
                    fg="red"
                )

                return

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            resized = cv2.resize(img_rgb, (96,96))

            # Normalize
            resized = resized.astype(np.float32) / 255.0

            # Reshape
            resized = resized.reshape(1,96,96,3)

            # Predict
            prediction = model.predict(resized)

            confidence = prediction[0][0]

            # Prediction Logic
            if confidence > 0.5:

                result = "NORMAL"

                color = "green"

                confidence_score = confidence * 100

            else:

                result = "PNEUMONIA DETECTED"

                color = "red"

                confidence_score = (1 - confidence) * 100

            # Update Label
            result_label.config(
                text=f"{result}\nConfidence: {confidence_score:.2f}%",
                fg=color
            )

            # Display Image
            display_image = Image.open(file_path)

            display_image = display_image.resize((350,350))

            photo = ImageTk.PhotoImage(display_image)

            image_label.config(image=photo)

            image_label.image = photo

        except Exception as e:

            result_label.config(
                text=f"Error:\n{str(e)}",
                fg="red"
            )

# =========================
# UPLOAD BUTTON
# =========================

upload_button = tk.Button(
    root,
    text="Upload X-Ray Image",
    command=upload_image,
    font=("Arial", 16, "bold"),
    bg="blue",
    fg="white",
    padx=20,
    pady=10
)

upload_button.pack(pady=20)

# =========================
# FOOTER
# =========================

footer = tk.Label(
    root,
    text="AI Based Pneumonia Detection",
    font=("Arial", 12),
    bg="white",
    fg="gray"
)

footer.pack(side="bottom", pady=10)

# =========================
# RUN APPLICATION
# =========================

root.mainloop()