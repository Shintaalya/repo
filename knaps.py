import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression

# Fungsi untuk mengambil model
def load_model():
    with open('model.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        X_train_expanded = model_data['X_train_expanded']
        y_train_mean = model_data['y_train_mean']
        y_train_std = model_data['y_train_std']
        best_X_train = model_data['best_X_train']
        best_y_train = model_data['best_y_train']
    return model, X_train_expanded, y_train_mean, y_train_std, best_X_train, best_y_train

# Fungsi untuk mengubah fitur input
def expand_input_features(data, best_X_train):
    normalized_data = (data - np.mean(best_X_train, axis=0)) / np.std(best_X_train, axis=0)
    expanded_data = PolynomialFeatures(degree=2).fit_transform(normalized_data)
    return expanded_data

# Fungsi untuk mengembalikan prediksi menjadi nilai semula
def denormalize_data(data, y_train_mean, y_train_std):
    denormalized_data = (data * y_train_std) + y_train_mean
    return denormalized_data

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.sidebar.title("Menu")
    choose = option_menu("Linear Regression (Polynomial)", ["Home", "Dataset", "Prepocessing", "Predict", "Help"],
                         icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "10A19D"},
                             "icon": {"color": "blue", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                          "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#00FFFF"},
                         })

    if choose == 'Home':
        st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
        logo = Image.open('makam1.jpg')
        st.image(logo, use_column_width=True, caption='Rumah di Jaksel')
        st.write('<p style = "text-align: justify;">Rumah merupakan salah satu kebutuhan pokok manusia, selain sandang dan pangan, rumah juga berfungsi sebagai tempat tinggal dan berfungsi untuk melindungi dari gangguan iklim dan makhluk hidup lainnya. Tak kalah buruknya dengan emas, rumah pun bisa dijadikan sebagai sarana investasi masa depan karena pergerakan harga yang berubah dari waktu ke waktu, dan semakin banyak orang yang membutuhkan hunian selain kedekatan dengan tempat kerja, pusat perkantoran dan pusat bisnis, transportasi. dll tentunya akan cepat mempengaruhi harga rumah tersebut.</p>', unsafe_allow_html=True)
        st.write('<p style = "text-align: justify;">Dalam proyek ini, kami mengembangkan sebuah sistem untuk memprediksi harga rumah berdasarkan parameter luas tanah dan luas bangunan, dan output yang dihasilkan adalah prediksi harga rumah. Kami menggunakan metode regresi linear dengan fitur ekspansi (expand feature) dan melatih model menggunakan metode Stochastic Gradient Descent. Untuk mengevaluasi model, kami menggunakan metrik MSE, RMSE, dan R (Square).Diharapkan dengan adanya sistem ini, dapat membantu dalam memprediksi harga rumah sesuai dengan luas tanah dan luas bangunan yang diinginkan.</p>', unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("Dr. Indah Agustien Siradjuddin, S.Kom., M.Kom")

    elif choose == 'Dataset':
        st.markdown('<h1 style = "text-align: center;"> Data Harga Rumah </h1>', unsafe_allow_html=True)
        df = pd.read_csv('https://raw.githubusercontent.com/Shintaalya/repo/main/HARGA%20RUMAH%20JAKSEL.csv')
        df
        st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #00FFFF; padding: 30px; border-radius: 20px;"><p>Dataset ini diambil dari kaggle.com</p><li><i><b>HARGA</b></i> = harga dari rumah</li><li><i><b>LT</b></i> = Jumlah Luas Tanah</li><li><i><b>LB</b></i> = Jumlah Luas Bangunan</li><li><i><b>JKT</b></i> = Jumlah Kamar Tidur</li><li><i><b>JKM</b></i> = Jumlah Kamar Mandi</li><li><i><b>GRS</b></i> = Ada / Tidak Ada</li></ol>',
                     unsafe_allow_html=True)

    elif choose == 'Prepocessing':
        st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
        st.write("Dari 7 Fitur")
        logo = Image.open('dataset.png')
        st.image(logo, caption='')
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("Diseleksi menjadi 2 Fitur")
        logo = Image.open('dataset2.png')
        st.image(logo, caption='')
        st.write("Berdasarkan garis lurus atau linearnya")
        logo = Image.open('dataset3.png')
        st.image(logo, caption='')

    elif choose == 'Predict':
        st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
        logo = Image.open('eror.png')
        st.image(logo, caption='')

        # Memuat model dan data terkait
        model, X_train_expanded, y_train_mean, y_train_std, best_X_train, best_y_train = load_model_data()

        # Fungsi untuk mem-normalisasi data input
        def normalize_input_data(data):
            normalized_data = (data - np.mean(best_X_train, axis=0)) / np.std(best_X_train, axis=0)
            return normalized_data

        # Fungsi untuk memperluas fitur input
        def expand_input_features(data):
            normalized_data = normalize_input_data(data)
            expanded_data = model.expand_features(normalized_data, degree=2)
            return expanded_data

        # Fungsi untuk mengembalikan data yang dinormalisasi
        def denormalize_data(data):
            denormalized_data = (data * y_train_std) + y_train_mean
            return denormalized_data

        # Kode aplikasi Streamlit
        def main():
            st.title('Prediksi Harga Rumah')

            # Form input
            input_data_1 = st.text_input('Luas Tanah', '100')
            input_data_2 = st.text_input('Luas Bangunan', '200')

            # Periksa apakah nilai input bersifat numerik
            if not input_data_1.isnumeric() or not input_data_2.isnumeric():
                st.error('Please enter numeric values for the input features.')
                return

            # Konversi nilai input ke float
            input_feature_1 = float(input_data_1)
            input_feature_2 = float(input_data_2)

            # Normalisasi dan perluas fitur input
            input_features = np.array([[input_feature_1, input_feature_2]])
            expanded_input = expand_input_features(input_features)

            # Melakukan prediksi
            normalized_prediction = model.predict(expanded_input)
            prediction = denormalize_data(normalized_prediction)

            # Menampilkan prediksi
            st.subheader('Hasil Prediksi')
            st.write(prediction[0])

        if __name__ == "__main__":
            main()

    elif choose == 'Help':
        st.markdown('<h1 style="text-align: center;"> Panduan : </h1><ol type="1" style="text-align: justify; background-color: #00FFFF; padding: 30px; border-radius: 20px;"><li><i><b>Cara View Dataset</b></i> <ol type="a"><li>Masuk ke sistem</li><li>Pilih menu dataset</li></ol></li><li><i><b>Cara Prediksi Harga</b></i> <ol type="a"><li>Pilih menu predict</li><li>Pilih LT dan LB</li><li>Klik tombol prediksi</li></ol></li></ol>',
                     unsafe_allow_html=True)

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
