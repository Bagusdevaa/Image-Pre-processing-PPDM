import streamlit as st
import cv2
import modules
from modules import modul
import pandas as pd
import matplotlib.pyplot as plt

# Menghilangkan warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Buat Object dari class buatan di file modul.py
proces = modul.ImagePreprocesing()

# Nomor image yang didapat setiap folder
img_listNo = [int(i) for i in range(607, 610)]


# Title
st.title("Image Pre-Processing PPDM")
st.write(
    '''
    Tugas kali ini adalah mengerjakan image pre-processing yang memiliki beberapa process
, yaitu analisis warna (konversi pixel ke matrix, colour histogram, dan first 
order statistics) dan analisis tekstur (GLCM, tekstur histogram , dan second order statistics)
    '''
)
st.markdown(
    """
    <style>
    .watermark {
        padding-bottom: 15px;
        font-size: 18px;
        color: gray;
        z-index: 99999 !important;
    }
    </style>
    <div class='watermark'>Created by Deva (2108561013)</div>
    """,
    unsafe_allow_html=True
)
# Title END
#################################################################
# IMPORT LIBRARY
st.header("Import Library")
st.code(
    '''
    import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
    '''
)
st.write("Membuat list dari nomor image yang didapatkan dan membuat path dari lokal file komputer kita")
st.code(
    '''
    # path
PATH = Path().cwd().parent.parent
DDIR = PATH / 'tugas-2' / 'assets'

# List nomor
img_listNo = [int(i) for i in range(607, 610)]
    
    '''
)
# IMPORT LIBRARY END

st.header("Colour Analysis")
# Gambar ke pixel matrix
st.subheader("Menampilkan proses konversi dari image ke pixel matrix")
with st.expander("Image to Pixel Matrix"):
    kategori_selected = st.selectbox(
        'Pilih Kategori',
        ('Happy','Neutral','Sad'),
        key='pixelMatixBox'
    )
    # Kategori Happy
    if kategori_selected == 'Happy':
        st.code(
            '''
            # Konversi ke matrix pixel
save = []
for j in img_listNo:
    pixel = cv2.imread(f'{DDIR}/happy/happy-0{j}.jpg', 0)
    save.append(pixel)
for i in range(3):
    print(f'Happy-{img_listNo[i]}')
    print(pd.DataFrame(save[i]))
            '''
        )
        for i in range(len(img_listNo)):
            st.write(f'Happy-{img_listNo[i]}')
            st.dataframe(proces.PixelToMatrix(index=i, imgList=img_listNo, kategori='happy'))
    # Kategori Netral
    elif kategori_selected == 'Neutral':
        st.code(
            '''
            # Konversi ke matrix pixel
save = []
for j in img_listNo:
    pixel = cv2.imread(f'{DDIR}/neutral/neutral-0{j}.jpg', 0)
    save.append(pixel)
for i in range(3):
    print(f'neutral-{img_listNo[i]}')
    print(pd.DataFrame(save[i]))
            '''
        )
        for i in range(len(img_listNo)):
            st.write(f'neutral-{img_listNo[i]}')
            st.dataframe(proces.PixelToMatrix(index=i, imgList=img_listNo, kategori='neutral'))
    # Kategori Sad
    elif kategori_selected =="Sad":
        st.code(
            '''
            # Konversi ke matrix pixel
save = []
for j in img_listNo:
    pixel = cv2.imread(f'{DDIR}/sad/sad-0{j}.jpg', 0)
    save.append(pixel)
for i in range(3):
    print(f'sad-{img_listNo[i]}')
    print(pd.DataFrame(save[i]))
            '''
        )
        for i in range(len(img_listNo)):
            st.write(f'sad-{img_listNo[i]}')
            st.dataframe(proces.PixelToMatrix(index=i, imgList=img_listNo, kategori='sad'))
# Gambar ke pixel matrix END
#########################################################
# Colour Histogram
st.subheader("Menampilkan proses membuat colour histogram dari setiap gambar")
with st.expander('Colour Histogram'):
    kategori_selected = st.selectbox(
        'Pilih Kategori',
        ('Happy','Neutral','Sad'),
        key='colourHistogramBox'
    )
    # Happy Select Box
    if kategori_selected == 'Happy':
        st.code(
            '''
            # Looping untuk menampilkan setiap gambar
for img in range(len(img_listNo)):
    # Size canvas visualisasi
    plt.figure(figsize=(30,4))

    # Read gambar grayscale, lgsg berubah jadi pixel matriks
    img = io.imread(f'{DDIR}/happy/happy-0{img_listNo[img]}.jpg')

    # Pembuatan colour histogram
    histogram, _ = np.histogram(img, bins=np.arange(0, 257))

    # Menampilkan hasil
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.bar(np.arange(0, 256), histogram)
    plt.title('Colour Histogram'), plt.xlabel('Bins'), plt.ylabel('Num of Pixel')
    plt.show()
            '''
        )
        for index in range(len(img_listNo)):
            plot = proces.ColourHistogram(imgList=img_listNo, kagetori="happy",index=index)
            st.pyplot(plot)
    # KATEGORI NETRAL
    elif kategori_selected == 'Neutral':
        st.code(
            '''
            # Looping untuk menampilkan setiap gambar
for img in range(len(img_listNo)):
    # Size canvas visualisasi
    plt.figure(figsize=(30,4))

    # Read gambar grayscale, lgsg berubah jadi pixel matriks
    img = io.imread(f'{DDIR}/neutral/neutral-0{img_listNo[img]}.jpg')

    # Pembuatan colour histogram
    histogram, _ = np.histogram(img, bins=np.arange(0, 257))

    # Menampilkan hasil
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.bar(np.arange(0, 256), histogram)
    plt.title('Colour Histogram'), plt.xlabel('Bins'), plt.ylabel('Num of Pixel')
    plt.show()
            '''
        )
        for index in range(len(img_listNo)):
            plot = proces.ColourHistogram(imgList=img_listNo, kagetori="neutral",index=index)
            st.pyplot(plot)
    # KATEGORI SAD
    elif kategori_selected =='Sad':
        st.code(
            '''
            # Looping untuk menampilkan setiap gambar
for img in range(len(img_listNo)):
    # Size canvas visualisasi
    plt.figure(figsize=(30,4))

    # Read gambar grayscale, lgsg berubah jadi pixel matriks
    img = io.imread(f'{DDIR}/sad/sad-0{img_listNo[img]}.jpg')

    # Pembuatan colour histogram
    histogram, _ = np.histogram(img, bins=np.arange(0, 257))

    # Menampilkan hasil
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.bar(np.arange(0, 256), histogram)
    plt.title('Colour Histogram'), plt.xlabel('Bins'), plt.ylabel('Num of Pixel')
    plt.show()
            '''
        )
        for index in range(len(img_listNo)):
            plot = proces.ColourHistogram(imgList=img_listNo, kagetori="sad",index=index)
            st.pyplot(plot)

# Colour Histogram END
st.subheader("Menampilkan proses first order statistics")
###############################################################
# First Order Statistics
with st.expander("First Order Statistics"):
    kategori_selected = st.selectbox(
    'Pilih Kategori',
    ('Happy','Neutral','Sad'),
    key='firstOrderBox'
    )
    # KATEGORI HAPPY
    if kategori_selected == "Happy":
        st.code(
            '''
            save = {}
for i in img_listNo:
    img = cv2.imread(f'{DDIR}/happy/happy-0{i}.jpg', 0)
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
pd.DataFrame.from_dict(save, orient='index', columns=['mean', 'variance', 'std_dev', 'skewness', 'kurtosis'])
            '''
        )
        st.dataframe(proces.FirstOrderStatistics(imgList=img_listNo,kategori='happy'))
    # KATEGORI NETRAL
    elif kategori_selected == 'Neutral':
        st.code(
            '''
            save = {}
for i in img_listNo:
    img = cv2.imread(f'{DDIR}/neutral/neutral-0{i}.jpg', 0)
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
    save[f'neutral-0{i}'] = [mean,variance,std_dev,skewness,kurtosis]

# Return hasil dalam bentuk dataframe
pd.DataFrame.from_dict(save, orient='index', columns=['mean', 'variance', 'std_dev', 'skewness', 'kurtosis'])
            '''
        )
        st.dataframe(proces.FirstOrderStatistics(imgList=img_listNo,kategori='neutral'))
        # KATEGORI SAD
    elif kategori_selected == "Sad":
        st.code(
            '''
            save = {}
for i in img_listNo:
    img = cv2.imread(f'{DDIR}/sad/sad-0{i}.jpg', 0)
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
    save[f'sad-0{i}'] = [mean,variance,std_dev,skewness,kurtosis]

# Return hasil dalam bentuk dataframe
pd.DataFrame.from_dict(save, orient='index', columns=['mean', 'variance', 'std_dev', 'skewness', 'kurtosis'])
            '''
        )
        st.dataframe(proces.FirstOrderStatistics(imgList=img_listNo,kategori='sad'))
# First Order Statistics END
#######################################################################

# Texture Analisis
st.header("Texture Analysis")

########################################################################
# GLCM
st.subheader("Menampilkan matrix GLCM")
with st.expander("GLCM"):
    st.code(
        '''
        def glcm(image, d, theta):
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

def normalize_glcm(glcm_matrix):
    # Hitung jumlah total nilai pada matriks
    total = np.sum(glcm_matrix)

    # Normalisasi matriks dengan membagi setiap nilai dengan total
    glcm_norm = glcm_matrix / total

    return glcm_norm


def glcm_features(glcm_matrix):
    # Hitung fitur
    contrast = np.sum(np.square(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1])) * glcm_matrix)
    dissimilarity = np.sum(np.abs(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1])) * glcm_matrix)
    homogeneity = np.sum(glcm_matrix / (1 + np.square(np.arange(glcm_matrix.shape[0]) - np.arange(glcm_matrix.shape[1]))))
    energy = np.sum(np.square(glcm_matrix))
    correlation = np.sum((glcm_matrix * (np.outer(np.arange(glcm_matrix.shape[0]), np.arange(glcm_matrix.shape[1])) - np.mean(glcm_matrix))) / (np.std(glcm_matrix) * np.std(np.outer(np.arange(glcm_matrix.shape[0]), np.arange(glcm_matrix.shape[1])))))

    return contrast, dissimilarity, homogeneity, energy, correlation


# Contoh penggunaan
# untuk menyimpan hasil fitur dari kesembilan gambar
features_list = {}
kat = ['happy','neutral','sad']
index = 0
# Loop untuk membaca setiap gambar dan menghitung fitur GLCM
for j in range(3):
    for k in range(607, 610):
        img_path = f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg'
        img = cv2.imread(img_path, 0)
        glcm_matrix = glcm(img, 1, 0)
        glcm_norm = normalize_glcm(glcm_matrix)
        contrast, dissimilarity, homogeneity, energy, correlation = glcm_features(glcm_norm)
        
        # Menyimpan hasil fitur ke dalam list
        features_list[f'{kat[j]}-{k}'] = [contrast, dissimilarity, homogeneity, energy, correlation]
    index += 1
# Konversi list hasil fitur ke dalam dataframe
df = pd.DataFrame.from_dict(features_list,orient="index" ,columns=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'])
df
        '''
    )
    # untuk menyimpan hasil fitur dari kesembilan gambar
    features_list = {}
    kat = ['happy','neutral','sad']
    index = 0
    # Loop untuk membaca setiap gambar dan menghitung fitur GLCM
    for j in range(3):
        for k in range(607, 610):
            img_path = f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg'
            img = cv2.imread(img_path, 0)
            glcm_matrix = proces.glcm(img, 1, 0)
            glcm_norm = proces.normalize_glcm(glcm_matrix)
            contrast, dissimilarity, homogeneity, energy, correlation = proces.glcm_features(glcm_norm)
            
            # Menyimpan hasil fitur ke dalam list
            features_list[f'{kat[j]}-{k}'] = [contrast, dissimilarity, homogeneity, energy, correlation]
        index += 1
        # Konversi list hasil fitur ke dalam dataframe
    df = pd.DataFrame.from_dict(features_list,orient="index" ,columns=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'])
    st.dataframe(df)
# GLCM END
#######################################################################################

# Texture Histogram
st.subheader("Menampilkan proses menampilkan texture histogram")
with st.expander("Texture Histogram"):
    st.code(
        '''
        def texture_histogram(glcm_matrix, num_bins=256):
    # Hitung histogram dengan membagi matrix GLCM ke dalam num_bins bin
    hist, _ = np.histogram(glcm_matrix, bins=num_bins)

    # Normalisasi histogram dengan membagi setiap nilai dengan total jumlah elemen dalam matrix GLCM
    total = np.sum(glcm_matrix)
    hist_norm = hist / total

    return hist_norm

# Output
for j in range(len(img_listNo)):
    for k in img_listNo:
        img_path = f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg'
        img = cv2.imread(img_path, 0)
        glcm_matrix = glcm(img, 2, 135)
        glcm_norm = normalize_glcm(glcm_matrix)
        hist = texture_histogram(glcm_norm)

        # Visualisasi histogram
        plt.plot(hist),plt.title(f"{kat[j]}-0{k}")
        plt.show()
        '''
    )
    for j in range(len(img_listNo)):
        for k in img_listNo:
            img_path = f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg'
            img = cv2.imread(img_path, 0)
            glcm_matrix = proces.glcm(img, 2, 135)
            glcm_norm = proces.normalize_glcm(glcm_matrix)
            hist = proces.texture_histogram(glcm_norm)

            # Visualisasi histogram
            plt.plot(hist),plt.title(f"{kat[j]}-0{k}");
            st.pyplot(plt.show());
# Texture Histogram END
##############################################################################
#Second order
st.subheader("Menampilkan second order statistics")
with st.expander("Second Order Statistics"):
    st.code(
        '''
        def calculate_second_order_stats(glcm_matrix):
    # Hitung second order statistics dari matrix GLCM
    contrast = np.sum(glcm_matrix * np.square(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float)))
    dissimilarity = np.sum(glcm_matrix * np.abs(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float)))
    homogeneity = np.sum(glcm_matrix / (1 + np.square(np.arange(glcm_matrix.shape[0], dtype=np.float) - np.arange(glcm_matrix.shape[1], dtype=np.float))))
    energy = np.sum(np.square(glcm_matrix))
    correlation = np.sum(glcm_matrix * (np.arange(glcm_matrix.shape[0], dtype=np.float) - np.mean(np.arange(glcm_matrix.shape[0], dtype=np.float))) \
                        * (np.arange(glcm_matrix.shape[1], dtype=np.float) - np.mean(np.arange(glcm_matrix.shape[1], dtype=np.float)))) \
                        / (np.std(np.arange(glcm_matrix.shape[0], dtype=np.float)) * np.std(np.arange(glcm_matrix.shape[1], dtype=np.float)))

    return contrast, dissimilarity, homogeneity, energy, correlation

for j in range(len(img_listNo)):
    for k in img_listNo:
        # Load citra Lena
        img = cv2.imread(f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg', 0)

        # Hitung matrix GLCM
        glcm_matrix = glcm(img, 1, 0)

        # Normalisasi GLCM
        glcm_norm = normalize_glcm(glcm_matrix)

        # Hitung second order statistics
        contrast, dissimilarity, homogeneity, energy, correlation = calculate_second_order_stats(glcm_norm)

        print(f"{kat[j]}-0{k}")
        print(f"Contrast: {contrast}")
        print(f"Dissimilarity: {dissimilarity}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Energy: {energy}")
        print(f"Correlation: {correlation}")
        print(30*"~")
        '''
    )
    for j in range(len(img_listNo)):
        for k in img_listNo:
            # Load citra Lena
            img = cv2.imread(f'D:/KULIAH/Mata Kuliah/semester-4/PPDM/tugas-2/assets/{kat[j]}/{kat[j]}-0{k}.jpg', 0)

            # Hitung matrix GLCM
            glcm_matrix = proces.glcm(img, 1, 0)

            # Normalisasi GLCM
            glcm_norm = proces.normalize_glcm(glcm_matrix)

            # Hitung second order statistics
            contrast, dissimilarity, homogeneity, energy, correlation = proces.calculate_second_order_stats(glcm_norm)

            st.write(f"{kat[j]}-0{k}")
            st.write(f"Contrast: {contrast}")
            st.write(f"Dissimilarity: {dissimilarity}")
            st.write(f"Homogeneity: {homogeneity}")
            st.write(f"Energy: {energy}")
            st.write(f"Correlation: {correlation}")
            st.write(30*"~")

#Second order END