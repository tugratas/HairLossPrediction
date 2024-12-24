import os
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model

# Model
model_path = 'resnet50v2_modelOverlayed95.h5'
model = load_model(model_path)

# Sınıf etiketleri (modelin eğitildiği sırayla)
class_labels = ['class1', 'class2', 'class3', 'class4']

# Klasör yolu
folder_path = './test/masks'  # Klasörün yolu

# Klasördeki tüm dosyaları listele
file_list = os.listdir(folder_path)

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)

    # Resim yükleme ve ön işleme
    img = image.load_img(file_path, target_size = (224, 224)) 
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis = 0)  
    x = x / 255.0  

    # Tahmin yap
    predictions = model.predict(x)

    # Tahminin sınıf etiketini al
    predicted_class = np.argmax(predictions, axis = 1)
    predicted_label = class_labels[predicted_class[0]]

    # Her sınıfın ağırlıklarını tanımlıyoruz (kök sayısı üzerinden tahmini şekilde)
    class_weights = [300, 3000, 7000, 20000]

    # Ağırlıklı ortalama hesaplıyoruz
    weighted_average = sum(predictions[0][i] * class_weights[i] for i in range(len(predictions[0])))

    # Çıkan sonuca göre bir mesaj yazdırıyoruz.
    if weighted_average < 1000:
        print("Saç ekimi yapılmasına gerek yoktur.")
    elif weighted_average > 7500:
        print("Saç ekim işlemi gerçekleştirilemez.")
    else:
        print(f"Ekilmesi gereken tahmini greft sayısı: {int(weighted_average)}")

    def percentage_calculator(predictions):
        class_count = len(predictions[0])
        percentage_guess = [round(rate * 100, 2) for rate in predictions[0]]
        classes = [f"Sınıf {i + 1}" for i in range(class_count)]
        return dict(zip(classes, percentage_guess))


    # Tahminleri yüzde cinsinden elde et
    percentage_guess = percentage_calculator(predictions)
    print("Tahminler (Yüzde):", percentage_guess)

    # Tahmin sonucunu yazdır
    print("Dosya Adı:", file_name)
    print("Tahmin Edilen Sınıf Etiketi:", predicted_label)
    print("---------------------------------------------")
