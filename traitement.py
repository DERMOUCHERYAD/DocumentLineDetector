import numpy as np
import matplotlib.image as mpim
import statistics
from matplotlib import pyplot as plt
import random
import cv2 as cv


def bords(img):
    #https://fr.wikipedia.org/wiki/Filtre_de_Sobel#:~:text=En%20traitement%20d'image%20%2C%20le,contours%20sont%20mis%20en%20exergue.
    #https://stackoverflow.com/questions/51167768/sobel-edge-detection-using-opencv

    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    ims = cv.resize(grad, (640, 360))
    cv.imshow('Edges', ims)
    cv.waitKey(0)

def k_means(img, k):

    centroids = tuple((random.randrange(0,255), random.randrange(0,255)))

    x = 0
    while x < k:

        classe1_masque = abs(centroids[0] - img) < abs(centroids[1] - img)
        classe2_masque = abs(centroids[0] - img) >= abs(centroids[1] - img)

        classe1 = img[classe1_masque]
        classe2 = img[classe2_masque]

        if classe1.size > 0 and classe2.size > 0:
            nouv_centroid1 = statistics.mean(classe1)
            nouv_centroid2 = statistics.mean(classe2)

            if centroids[0] == nouv_centroid1 and centroids[1] == nouv_centroid2:
                break
            else:
                centroids = (nouv_centroid1, nouv_centroid2)

        x += 1

    return centroids

def seuillage(img, centroids):
    # Conversion de l'image en float pour éviter les problèmes d'overflow
    img_float = img.astype(np.float32)
    
    # Utilisation d'une approche vectorisée pour le seuillage
    diff0 = np.abs(img_float - centroids[0])
    diff1 = np.abs(img_float - centroids[1])
    
    # Création d'une image binaire : 1 si la différence par rapport au premier centroid est supérieure, sinon 0
    img2 = np.where(diff0 > diff1, 1, 0)
    
    return img2


def binarisation(img):
    # Si l'image est en couleur, on la convertit en niveaux de gris
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Optionnel : appliquer un flou gaussien pour réduire le bruit
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    
    # Utilisation de la fonction k_means pour obtenir deux centroids.
    # Le paramètre 10 ici correspond au nombre maximum d'itérations pour la convergence.
    centroids = k_means(blurred, 10)
    
    # Binarisation de l'image en utilisant la fonction seuillage basée sur les centroids
    bin_img = seuillage(blurred, centroids)
    
    return bin_img



def histogramme_projection(img_binaire):
    """
    Calcule l'histogramme de projection sur l'axe vertical
    en sommant les valeurs de pixels sur chaque ligne.
    """
    return np.sum(img_binaire, axis=1)

# Liste des images à traiter
liste_images = ['doc1.jpg', 'doc2.jpg', 'doc3.jpg', 'doc4.jpg']

for image_name in liste_images:
    # Chargement de l'image
    img = cv.imread(image_name)
    if img is None:
        print(f"Erreur lors du chargement de {image_name}")
        continue

    # Applique la binarisation à l'image
    bin_img = binarisation(img)  # Votre fonction binarisation doit être définie

    # Calcul de l'histogramme de projection sur l'axe vertical
    hist = histogramme_projection(bin_img)

    # Affichage avec matplotlib : utilisation de subplots pour afficher l'image et son histogramme
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Affichage de l'image binarisée
    axs[0].imshow(bin_img, cmap='gray')
    axs[0].set_title(f'Binarisation de {image_name}')
    axs[0].axis('off')
    
    # Affichage de l'histogramme de projection
    axs[1].plot(hist)
    axs[1].set_title(f'Histogramme de projection pour {image_name}')
    axs[1].set_xlabel("Position verticale (ligne)")
    axs[1].set_ylabel("Nombre de pixels (texte)")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()


