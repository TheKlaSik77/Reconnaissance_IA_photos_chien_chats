Le fichier dataset contient toutes les photos nécessaires au bon fonctionnement du programme.

creation_reseau_neurones.py est le programme qui après execution, génère le fichier testChienChat.h5 -> Le réseau de neurone qu'il est possible de charger dans le fichier test_sur_single_prediction.py. Ce dernier est le code qui utilise notre réseau de neurones afin de déterminer si les 2 photos + 1 que j'ai rajouté (voir plus bas pourquoi) sont des chiens ou des chats.

Je me suis permis de rajouter une image plus ambigue sur laquelle un enfant tient un chat. Cette image étant détéctée à 56% comme un chien par le réseau de neurones, j'ai donc pris la liberté de modifier le paramètre dans le Layer 4 de Conv2D en le passant de 128 à 256 afin d'être plus précis dans l'analyse des caractèristiques. Après recréation du réseau de neurones, ce pourcentage passe à 9% de chien, soit l'image 3 est à 91% un chat.
