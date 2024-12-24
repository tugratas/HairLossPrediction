import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import dlib

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

# Dlib yüz dedektörü ve landmark tespiti modelini yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# modeli yükle
model_path = './saclarin_segmentasyon_modeli.h5'
model = tf.keras.models.load_model(model_path)

# segmentasyon yapılacak görüntülerin klasörü
image_folder = './test/imgs'
segmented_folder = './test/masks'

# klasördeki her görüntü için döngü
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_file)

        # Görüntüyü yükle ve 256x256 boyutuna yeniden boyutlandır
        preprocessed_image = load_and_preprocess_image(image_path)

        # model üzerinde tahmin yapma
        prediction = model.predict(preprocessed_image)

        # tahmin işleme ve sonuçları kaydetme
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Yüz landmarklarını tespit etme
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Maskenin altındaki kısmı siyah yapma
        segmented_image = postprocess_prediction(prediction, original_image)
        segmented_image = mask_below_nose(segmented_image, landmarks)

        # # Orjinal resmi %0 opaklıkta segmentasyon sonucu ile birleştirme ve kaydetme
        overlayed_image = overlay_images(original_image, segmented_image, alpha=0.5)
        overlayed_image_save_path = os.path.join(segmented_folder, image_file)
        cv2.imwrite(overlayed_image_save_path, overlayed_image)

        # Segmentasyon sonucunu direkt olarak kaydetme
        #segmented_image_save_path = os.path.join(segmented_folder, "segmented_" + image_file)
        #cv2.imwrite(segmented_image_save_path, segmented_image)

print("Tüm segmentasyonlar tamamlandı ve kaydedildi.")