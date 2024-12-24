import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Veri yükleme ve işleme fonksiyonu
def load_dataset(image_dir, mask_dir, image_size=(256, 256)):
    images = []  # Görüntüleri tutacak liste
    masks = []  # Maskeleri tutacak liste

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)  # .jpg uzantısı korumak için
        img = load_img(img_path, target_size=image_size)
        img = img_to_array(img) / 255.0
        images.append(img)

        mask = load_img(mask_path, target_size=image_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        mask = np.round(mask)  # Maskeleri binary forma dönüştür
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    return images, masks

# Unet modeli oluşturma
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # İlk katman
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    # İkinci katman
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same',
               kernel_initializer='he_normal')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    return x

def get_unet(input_img, n_filters=32, dropout=0.1, batchnorm=True):
    # Kontraksiyon yolu (aşağı doğru)
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    p5 = Dropout(dropout)(p5)

    c6 = conv2d_block(p5, n_filters=n_filters * 32, kernel_size=3, batchnorm=batchnorm)

    # Ekspansiyon yolu (yukarı doğru)
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c5], axis=3)
    c7 = conv2d_block(u7, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c4], axis=3)
    c8 = conv2d_block(u8, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c3], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u10 = UpSampling2D((2, 2))(c9)
    u10 = concatenate([u10, c2], axis=3)
    c10 = conv2d_block(u10, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u11 = UpSampling2D((2, 2))(c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = conv2d_block(u11, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # Çıktı katmanı
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Modeli eğitmek için parametreler
image_dir = './train/imgs'
mask_dir = './train/masks'
input_img = Input((256, 256, 3), name='img')

# Veri setini yükleme ve işleme
X, Y = load_dataset("./train/imgs", "./train/masks", image_size=(256, 256))

# Veri setini böleme
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Modeli oluşturma ve derleme
model = get_unet(input_img, n_filters=32, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Modeli eğitme
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('saclarin_segmentasyon_modeli.h5', verbose=1, save_best_only=True)
]

results = model.fit(X_train, Y_train, batch_size=32, epochs=20, callbacks=callbacks,
                    validation_data=(X_val, Y_val))
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save('saclarin_segmentasyon_modeli.h5')
