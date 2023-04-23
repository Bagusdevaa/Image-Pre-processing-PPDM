import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import streamlit as st

class ImagePreprocesing:
    def __init__(self):
        PATH = Path().cwd().parent.parent
        self.DDIR = PATH / 'tugas-2' / 'assets'

    # Pixel to Matrix
    def PixelToMatrix(self, index, imgList, kategori):
        self.save = []
        for j in imgList:
            pixel = cv2.imread(f'{self.DDIR}/{kategori}/{kategori}-0{j}.jpg', 0)
            self.save.append(pixel)

        return pd.DataFrame(self.save[index])
    
    # Color Histogram
    def ColourHistogram(self, imgList, kagetori, index):
        # Size canvas visualisasi
        plt.figure(figsize=(30,4))

        # Read gambar grayscale, lgsg berubah jadi pixel matriks
        img = cv2.imread(f'{self.DDIR}/{kagetori}/{kagetori}-0{imgList[index]}.jpg', 0)

        # Pembuatan colour histogram
        histogram, _ = np.histogram(img, bins=np.arange(0, 257))

        # Menampilkan hasil
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.bar(np.arange(0, 256), histogram)
        plt.title('Colour Histogram'), plt.xlabel('Bins'), plt.ylabel('Num of Pixel')
        return plt
    
    # First Order
    def FirstOrderStatistics(self, imgList, kategori):
        save = {}

        for i in imgList:
            img = cv2.imread(f'{self.DDIR}/{kategori}/{kategori}-0{i}.jpg', 0)
            # Hitung mean
            mean = np.mean(img)
            # Hitung variance
            variance = np.var(img)
            # Hitung standar deviasi
            std_dev = np.sqrt(variance)
            # Hitung skewness
            skewness = np.mean((img - mean)**3) / (std_dev**3)
            # Hitung kurtosis
            kurtosis = np.mean((img - mean)**4) / (std_dev**4)
            save[f'happy-0{i}'] = [mean,variance,std_dev,skewness,kurtosis]

        # Return hasil dalam bentuk dataframe
        return pd.DataFrame.from_dict(save, orient='index', columns=['mean', 'variance', 'std_dev', 'skewness', 'kurtosis'])
    
    # GLCM
    def glcm(self, image, d, theta):
        # Konversi ke grayscale jika belum grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Hitung matrix GLCM
        glcm_matrix = np.zeros((256, 256))

        # Loop untuk mengisi nilai matrix GLCM
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Tentukan koordinat piksel tujuan sesuai dengan nilai d dan theta
                if theta == 0:
                    x = i
                    y = j + d
                elif theta == 45:
                    x = i - d
                    y = j + d
                elif theta == 90:
                    x = i - d
                    y = j
                elif theta == 135:
                    x = i - d
                    y = j - d
                
                # Cek apakah koordinat piksel tujuan berada dalam range gambar
                if x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]:
                    glcm_matrix[image[i][j], image[x][y]] += 1

        return glcm_matrix.astype(int)

    def normalize_glcm(self,glcm_matrix):
        # Hitung jumlah total nilai pada matriks
        total = np.sum(glcm_matrix)

        # Normalisasi matriks dengan membagi setiap nilai dengan total
        glcm_norm = glcm_matrix / total

        return glcm_norm


    def glcm_features(self,glcm_matrix):
        # Hitung fitur
        contrast = np.sum(np.square(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1])) * glcm_matrix)
        dissimilarity = np.sum(np.abs(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1])) * glcm_matrix)
        homogeneity = np.sum(glcm_matrix / (1 + np.square(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1]))))
        energy = np.sum(np.square(glcm_matrix))
        correlation = np.sum((glcm_matrix * (np.outer(np.arange(glcm_matrix.shape[0]), np.arange(glcm_matrix.shape[1])) - np.mean(glcm_matrix))) / (np.std(glcm_matrix) * np.std(np.outer(np.arange(glcm_matrix.shape[0]), np.arange(glcm_matrix.shape[1])))))

        return contrast, dissimilarity, homogeneity, energy, correlation
    
    def texture_histogram(self,glcm_matrix, num_bins=256):
        # Hitung histogram dengan membagi matrix GLCM ke dalam num_bins bin
        hist, _ = np.histogram(glcm_matrix, bins=num_bins);

        # Normalisasi histogram dengan membagi setiap nilai dengan total jumlah elemen dalam matrix GLCM
        total = np.sum(glcm_matrix);
        hist_norm = hist / total;

        return hist_norm;

    def calculate_second_order_stats(self,glcm_matrix):
        # Hitung second order statistics dari matrix GLCM
        contrast = np.sum(glcm_matrix * np.square(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float)))
        dissimilarity = np.sum(glcm_matrix * np.abs(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float)))
        homogeneity = np.sum(glcm_matrix / (1 + np.square(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float))))
        energy = np.sum(np.square(glcm_matrix))
        correlation = np.sum(glcm_matrix * (np.arange(glcm_matrix.shape[0], dtype=np.float) - np.mean(np.arange(glcm_matrix.shape[0], dtype=np.float))) \
                            * (np.arange(glcm_matrix.shape[1], dtype=np.float) - np.mean(np.arange(glcm_matrix.shape[1], dtype=np.float)))) \
                            / (np.std(np.arange(glcm_matrix.shape[0], dtype=np.float)) * np.std(np.arange(glcm_matrix.shape[1], dtype=np.float)))

        return contrast, dissimilarity, homogeneity, energy, correlation