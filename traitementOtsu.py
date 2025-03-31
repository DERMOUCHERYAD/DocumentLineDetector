import numpy as np
import matplotlib.image as mpim
import statistics
from matplotlib import pyplot as plt
import random
import cv2 as cv

def bords(img):
    # Filtre de Sobel pour détecter les contours (références Wikipédia et StackOverflow)
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    ims = cv.resize(grad, (640, 360))
    cv.imshow('Edges', ims)
    cv.waitKey(0)

def k_means(img, max_iter):
    # Initialisation aléatoire de deux centroids pour une image en niveaux de gris
    centroids = (random.randrange(0, 255), random.randrange(0, 255))
    
    iter_count = 0
    while iter_count < max_iter:
        # Séparation des pixels en deux classes selon leur proximité aux centroids
        classe1_masque = np.abs(img - centroids[0]) < np.abs(img - centroids[1])
        classe2_masque = ~classe1_masque  # complémentaire
        
        classe1 = img[classe1_masque]
        classe2 = img[classe2_masque]
        
        if classe1.size > 0 and classe2.size > 0:
            nouv_centroid1 = statistics.mean(classe1)
            nouv_centroid2 = statistics.mean(classe2)
            
            if centroids[0] == nouv_centroid1 and centroids[1] == nouv_centroid2:
                break
            else:
                centroids = (nouv_centroid1, nouv_centroid2)
        iter_count += 1
    
    return centroids

def seuillage(img, centroids):
    # Conversion en float pour éviter les problèmes d'overflow (uint8)
    img_float = img.astype(np.float32)
    
    # Comparaison vectorisée avec les centroids
    diff0 = np.abs(img_float - centroids[0])
    diff1 = np.abs(img_float - centroids[1])
    
    # Création d'une image binaire : 1 si la différence avec le premier centroid est supérieure, sinon 0
    img2 = np.where(diff0 > diff1, 1, 0)
    return img2

def binarisation_kmeans(img):
    # Conversion en niveaux de gris si nécessaire
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Application d'un flou gaussien pour réduire le bruit
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    
    # Calcul des centroids par k-means (10 itérations maximum)
    centroids = k_means(blurred, 10)
    
    # Binarisation basée sur le seuillage par rapport aux centroids
    bin_img = seuillage(blurred, centroids)
    return bin_img

def binarisation_otsu(img):
    # Conversion en niveaux de gris si nécessaire
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Normalisation pour étendre la plage des intensités de 0 à 255
    norm_img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    
    # Application du seuillage global d’Otsu
    ret, bin_img = cv.threshold(norm_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Conversion en image binaire avec valeurs 0 et 1
    bin_img = bin_img // 255
    return bin_img, ret, norm_img

def histogramme_projection(img_binaire):
    """
    Calcule l'histogramme de projection sur l'axe vertical
    en sommant les valeurs de pixels sur chaque ligne.
    """
    return np.sum(img_binaire, axis=1)

def estimate_number_of_lines(hist, threshold_ratio=0.1):
    """
    Estime le nombre de lignes de texte à partir de l'histogramme.
    
    Le seuil est défini comme un pourcentage du pic maximal de l'histogramme.
    
    Returns:
        num_lines (int) : Nombre de segments (lignes) détectés.
        segments (list of tuples) : Liste des segments (début, fin) pour chaque ligne.
    """
    seuil = threshold_ratio * np.max(hist)
    num_lines = 0
    segments = []
    in_line = False
    start = 0

    for i, value in enumerate(hist):
        if value > seuil and not in_line:
            in_line = True
            start = i
        elif value <= seuil and in_line:
            in_line = False
            end = i - 1
            segments.append((start, end))
            num_lines += 1

    if in_line:
        segments.append((start, len(hist) - 1))
        num_lines += 1

    return num_lines, segments

# Liste des images à traiter
liste_images = ['doc1.jpg', 'doc2.jpg', 'doc3.jpg', 'doc4.jpg']

for image_name in liste_images:
    # Chargement de l'image
    img = cv.imread(image_name)
    if img is None:
        print(f"Erreur lors du chargement de {image_name}")
        continue

    # --- Méthode K-means ---
    bin_img_kmeans = binarisation_kmeans(img)
    hist_kmeans = histogramme_projection(bin_img_kmeans)
    num_lines_km, segments_km = estimate_number_of_lines(hist_kmeans, threshold_ratio=0.1)
    
    # --- Méthode Otsu ---
    bin_img_otsu, seuil_otsu, norm_img = binarisation_otsu(img)
    hist_otsu = histogramme_projection(bin_img_otsu)
    num_lines_otsu, segments_otsu = estimate_number_of_lines(hist_otsu, threshold_ratio=0.1)
    
    # Affichage avec matplotlib : comparaison sur 2 lignes (K-means et Otsu)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Méthode K-means
    axs[0, 0].imshow(bin_img_kmeans, cmap='gray')
    axs[0, 0].set_title(f'Binarisation K-means\n{image_name}')
    axs[0, 0].axis('off')
    
    axs[0, 1].plot(hist_kmeans, label="Histogramme")
    seuil_km = 0.2 * np.max(hist_kmeans)
    axs[0, 1].axhline(y=seuil_km, color='red', linestyle='--', label=f'Seuil = {seuil_km:.0f}')
    for (start, end) in segments_km:
        axs[0, 1].axvspan(start, end, color='green', alpha=0.3)
    axs[0, 1].set_title(f'Histogramme K-means\nNombre de lignes estimé = {num_lines_km}')
    axs[0, 1].set_xlabel("Position verticale (ligne)")
    axs[0, 1].set_ylabel("Nombre de pixels (texte)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Méthode Otsu
    axs[1, 0].imshow(bin_img_otsu, cmap='gray')
    axs[1, 0].set_title(f'Binarisation Otsu\n{image_name}\n(Seuil Otsu = {seuil_otsu:.0f})')
    axs[1, 0].axis('off')
    
    axs[1, 1].plot(hist_otsu, label="Histogramme")
    seuil_otsu_proj = 0.2 * np.max(hist_otsu)
    axs[1, 1].axhline(y=seuil_otsu_proj, color='red', linestyle='--', label=f'Seuil = {seuil_otsu_proj:.0f}')
    for (start, end) in segments_otsu:
        axs[1, 1].axvspan(start, end, color='green', alpha=0.3)
    axs[1, 1].set_title(f'Histogramme Otsu\nNombre de lignes estimé = {num_lines_otsu}')
    axs[1, 1].set_xlabel("Position verticale (ligne)")
    axs[1, 1].set_ylabel("Nombre de pixels (texte)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
