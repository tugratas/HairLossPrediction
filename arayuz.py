import cv2
import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import threading
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import dlib
from keras.preprocessing import image
from keras.models import load_model as keras_load_model

class CameraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("HairCheck AI")
        self.master.geometry("600x600")
        self.master.config(bg="#333")
        self.master.iconbitmap('icon.ico')

        self.image_label = tk.Label(self.master, bg="#222", fg="white")
        self.image_label.pack(pady=10)

        self.btn_capture = tk.Button(self.master, text="Capture", command=self.capture_image, bg="#444", fg="white", relief=tk.FLAT, padx=20, pady=5)
        self.btn_capture.pack(fill=tk.X, padx=10, pady=5)

        self.btn_run = tk.Button(self.master, text="Run", command=self.run_application, bg="#444", fg="white", relief=tk.FLAT, padx=20, pady=5)
        self.btn_run.pack(fill=tk.X, padx=10, pady=5)

        self.camera = cv2.VideoCapture(0)
        self.update_preview() 
        
        self.load_models_and_detectors()

    def load_models_and_detectors(self):
        # DLib kütüphanesi
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./modeller/shape_predictor_68_face_landmarks.dat")

        # Segmentasyon modeli
        self.segmentation_model = keras_load_model('./modeller/saclarin_segmentasyon_modeli.h5')

        # Sınıflandırma modeli
        self.classification_model = keras_load_model('./modeller/resnet50v2_modelOverlayed95.h5')

    def update_preview(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image.thumbnail((590, 590))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        self.master.after(10, self.update_preview) 

    def capture_image(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            frame_cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

            if not os.path.exists("test/imgs"):
                os.makedirs("test/imgs")
            cv2.imwrite("test/imgs/captured_image.jpg", frame_cropped)

            image = Image.open("test/imgs/captured_image.jpg")
            image.thumbnail((256, 256))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def display_image(self, path, parent, side):
        image = Image.open(path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(parent, image=photo, bg="#333", fg="white")
        label.image = photo
        label.pack(side=side, padx=10, pady=10)

    def run_application(self):
        threading.Thread(target=self.segment_and_classify).start()

    def segment_and_classify(self):
        def load_and_preprocess_image(image_path):
            img = load_img(image_path, target_size=(256, 256))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

        def postprocess_prediction(prediction, original_image):
            mask = prediction.squeeze()
            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
            segmented_image = np.zeros_like(original_image)
            segmented_image[mask == 1] = [255, 255, 255]
            return segmented_image

        def mask_below_nose(image, landmarks):
            nose_landmark = landmarks[30]
            nose_y = int(nose_landmark[1])
            image[nose_y:, :] = [0, 0, 0]
            return image

        def overlay_images(image, overlay, alpha=0.5):
            combined_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            return combined_image

        image_folder = './test/imgs'
        segmented_folder = './test/masks'

        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, image_file)

                preprocessed_image = load_and_preprocess_image(image_path)

                prediction = self.segmentation_model.predict(preprocessed_image)

                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                segmented_image = postprocess_prediction(prediction, original_image)
                segmented_image = mask_below_nose(segmented_image, landmarks)

                overlayed_image = overlay_images(original_image, segmented_image, alpha=0.5)
                overlayed_image_save_path = os.path.join(segmented_folder, image_file)
                cv2.imwrite(overlayed_image_save_path, overlayed_image)

        # Sınıflandırma sonuçları
        self.display_result()

    def display_result(self):
        new_window = Toplevel(self.master)
        new_window.title("HairCheck AI - Result")
        new_window.geometry("800x600")
        new_window.config(bg="#333")  # Set background color to dark
        new_window.iconbitmap('icon.ico')

        image_frame = tk.Frame(new_window, bg="#333")
        image_frame.pack(pady=10)

        # Orijinal fotoğraf
        original_image_path = "./test/imgs/captured_image.jpg"
        self.display_image(original_image_path, image_frame, side=tk.LEFT)

        # Segmente edilmiş fotoğraf
        segmented_image_path = "./test/masks/captured_image.jpg"
        self.display_image(segmented_image_path, image_frame, side=tk.LEFT)

        file_path = "./test/masks/captured_image.jpg"

        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        predictions = self.classification_model.predict(x)

        predicted_class = np.argmax(predictions, axis=1)[0]

        def percentage_calculator(predictions):
            class_count = len(predictions[0])
            percentage_guess = [round(rate * 100, 2) for rate in predictions[0]]
            classes = [f"Class {i + 1}" for i in range(class_count)]
            return dict(zip(classes, percentage_guess))

        percentage_guess = percentage_calculator(predictions)

        percentage_text = "Predictions (Percentage): " + ", ".join([f"{cls}: {pct}%" for cls, pct in percentage_guess.items()])
        percentage_label = tk.Label(new_window, text=percentage_text, bg="#333", fg="white", font=("Arial", 12))
        percentage_label.pack(pady=10)

        detailed_results = {
            0: {
                "title": "Class 1:",
                "description": "Your hair shows minimal hairline recession, and your hairline remains mostly intact. At this stage, your hair loss is very mild. Maintaining a healthy hair care routine and proper nutrition can help keep your hair in good condition."
            },
            1: {
                "title": "Class 2:",
                "description": "You may notice some recession in your hairline, and thinning might start to appear at the crown. At this stage, considering medical treatments or making lifestyle changes can help slow down or stop further hair loss."
            },
            2: {
                "title": "Class 3:",
                "description": "You likely experience significant thinning or complete loss of hair in the crown and frontal areas. Intensive treatment options such as hair transplants or other surgical methods may be suitable for you."
            },
            3: {
                "title": "Class 4:",
                "description": "You have experienced complete baldness. Many people in this category choose to embrace their bald look, though hair prostheses or other options are available if desired."
            }
        }

        if predicted_class in detailed_results:
            title = detailed_results[predicted_class]["title"]
            description = detailed_results[predicted_class]["description"]

            title_label = tk.Label(new_window, text=title, bg="#333", fg="white", font=("Arial", 16, "bold"))
            title_label.pack(pady=5)

            description_label = tk.Label(new_window, text=description, bg="#333", fg="white", font=("Arial", 14), wraplength=750, justify="left")
            description_label.pack(pady=0)

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
