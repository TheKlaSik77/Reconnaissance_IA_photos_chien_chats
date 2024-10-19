# Charger le modèle enregistré
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

# On charge le model
model = load_model('testChienChat.h5')

"""
On prépare les images tests :
    - On redimensionne l'image sur 150x150 (taille utilisée par le modèle)
    - On normalise en divisant les pixels par 255 pour avoir des pixels de 0 à 1
"""
def prepare_image(img_path):
    
    # Charger l'image et redimensionner à 150x150
    img = image.load_img(img_path, target_size=(150, 150))  
    
    # Convertir l'image en tableau numpy
    img_array = image.img_to_array(img)       

    # Ajouter une dimension pour correspondre à la forme (1, 150, 150, 3)          
    img_array = np.expand_dims(img_array, axis=0) 

    # Normaliser en divisant par 255          
    img_array /= 255.0    

    return img_array


# Tester le modèle avec de nouvelles images
img_array1 = prepare_image('./dataset/single_prediction/cat_or_dog_1.jpg')
img_array2 = prepare_image('./dataset/single_prediction/cat_or_dog_2.jpg')
img_array3 = prepare_image('./dataset/single_prediction/cat_or_dog_3.jpg')


prediction = model.predict(img_array1)
result = int(prediction > 0.5)
print("Chat" if result == 0 else "Chien")
print(f'\nPhoto 1 est a {prediction*100} % un Chien')

prediction2 = model.predict(img_array2)
result = int(prediction2 > 0.5)
print("Chat" if result == 0 else "Chien")
print(f'\nPhoto 2 est a {prediction2*100} % un Chien')

prediction3 = model.predict(img_array3)
result = int(prediction3 > 0.5)
print("Chat" if result == 0 else "Chien")
print(f'\nPhoto 3 est a {prediction3*100} % un Chien')


