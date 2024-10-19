import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import class_weight as cw
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping # type: ignore


# Pré-traite les images. Les normalisent en divisant leurs valeurs de pixels par 255, ce qui nous donne des pixels sur une plage de 0 à 1.
train = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)


# On charge les images d'entrainement (training_set) et de test (test_set), on redimensionne chaque image à 150x150. On crée des lots de 32 images à la fois (par batchs)
train_dataset = train.flow_from_directory("./dataset/training_set",target_size=(150,150),batch_size = 32,class_mode = 'binary')

test_dataset = test.flow_from_directory("./dataset/test_set",target_size=(150,150),batch_size = 32,class_mode = 'binary')


# Réseau de neurones (CNN = architecture du réseau convolutif) qui va traiter les images. 

model = Sequential()

"""
Les 4 premiers Layers vont utiliser :
    - des couches convolutives (Conv2D) pour extraire des caractèristiques visuelles des images telles que des bords,  des textures etc...

    - des couches de Max-Pooling qui permet de réduire la taille des images tout en conservant les infos les plus importantes.

La seule chose qui change pour ces 4 layers est l'augmentation du nombre de filtres (32->64->128). 
    - 32 filtres permettent d'extraire des caractèristiques simples comme les contours ou les bords de l'image (communes à la plupart des objets)
    - + de 64 filtres capture des caractèristiques plus complexes comme des motifs spécifiques ou des parties de l'objet (ex yeux, oreilles, textures plus détaillées). En augmentant le nb de filtres , on permet au modèle d'apprendre une plus grande diversité de motifs complexes.

Pourquoi ne pas toujours prendre un grand nombre de filtre ?

    - Ca coute cher en mémoire et temps de calcul
    - peut conduire à un surapprentissage -> Le modèle va chercher à apprendre des détails spécifiques aux images d'entrainement au lieu de généraliser correctement à de nouvelles images.
"""
# Convolutional layer and maxpool layer 1
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
# Convolutional layer and maxpool layer 2
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
# Convolutional layer and maxpool layer 3
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
# Convolutional layer and maxpool layer 4
# La valeur de base étant à 128, je l'ai mise à 256 afin de permettre d'être plus précis afin que la photo 3 que j'ai ajouté dans single prediction, soit bien détectée comme chat malgrès la présence de l'enfant
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))


"""
Sur les couches finales :
    - Flatten() va transformer la sortie 2D des convolutions en un vecteur 1D, ça signifie qu'on aura chaque pixel à la suite au lieu d'avoir une matrice de pixels

    - La couche Dense() avec 512 nerones applique une activation ReLU pour capturer des interactions complexes.

    - La couche Dense est une couche sigmoïde avec un seul neurone. Elle renvoie une probabilité entre 0 et 1 plus elle proche de 0 c'est un chien, et plus elle proche de 1 plus c'est un chat (ou inversement selon le code)

Pour le param nb_neurones : + de nerones = + de couts calculs et risques de surapprentissage.

Pour la couche de sortie on a généralement une couche Dense() avec un nb de nerones égal au nombre de classes (dans notre cas 1 car problème binaire (chat ou chien))
"""

# This layer flattens the resulting image array to 1D array
model.add(Flatten())
# Hidden layer with 512 neurons and Rectified Linear Unit activation function
model.add(Dense(650,activation='relu'))
# Output layer with single neuron which gives 0 for Cat or 1 for Dog
#Here we use sigmoid activation function which makes our model output to lie between
# 0 and 1
model.add(Dense(1,activation='sigmoid'))



# On utilise l'optimiseur 'adam', la fonction de perte 'binary_crossentropy' et la métrique d'évaluation 'accuracy' afin de suivre la précision pendant l'entrainement
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

"""
1/ L'optimiseur choisit la façon dont les poids du modèle sont ajustés pendant l'entrainement
    - Adam (pour Adaptive Moment Estimation) est l'un des optimisateurs les plus populaires en deep learning, car il combine les avantages des algo AdaGrad et RMSProp -> Apprentissage rapide et efficace
    - Il est rapide et adapté aux problèmes contenant bcp de données
    - Il ajuste les taux d'apprentissage insividuellement pour chaque paramètre en fonction de la première et deuxième moyenne exponentielle des gradients.

2/ La fonction de perte permet de calculer l'écart entre deux probas, ici entre la prédiction du modèle et classe réelle. On utilise la plus courante : binary_crossentropy.

3/ Les métriques permettent de suivre les performances du modèle dirant et parès l'entraînement. La métrique utilisée ici est 'accuracy' (ration d'exemples bien classés sur le total). 
Il est possible d'ajouter d'autres métriques (précision, recall, F1-score,etc...)
"""

"""
Ici on, on entraine le modèle en lui donnant:
    - le jeu de données d'entrainement : train_dataset
    - le nombre d'époque, c'est à dire le nombre de fois où le modèle va s'entraîner en traitant toutes les images.
    - le jeu de validation : test_dataset

Que fait la méthode fit ?

Celle ci va 
    - parcourir dix fois les données d'entraînement (epochs=10)
    - A chaque époque, il divise les données en lots (batches) et met à jour les poids du modèle après chaque lot
    - Après chaque époque, il évalue aussi le modèle sur le jeu de validation (test_dataset) pour vérifier si il fonctionne bien sur les données qu'il n'a jamais vu.
    - L'objet 'history' renvoyé contient l'évolution des métriques pour chaque époque.
"""
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)


# Récupérer les valeurs d'entraînement et de validation
acc = history.history['accuracy']        # Précision sur les données d'entraînement
val_acc = history.history['val_accuracy'] # Précision sur les données de validation

loss = history.history['loss']            # Perte sur les données d'entraînement
val_loss = history.history['val_loss']    # Perte sur les données de validation

# Création d'une plage d'epochs
epochs_range = range(1, len(acc) + 1)

# On sauvegarde le model
model.save('./testChienChat.h5')

# Affiche les classes pour valeurs
print(train_dataset.class_indices)

# Tracer la précision
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Tracer la perte
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
