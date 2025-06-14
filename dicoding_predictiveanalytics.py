# -*- coding: utf-8 -*-
"""Dicoding-PredictiveAnalytics.ipynb

Proyek Pertama : Predictive Analytics - Addin Hadi Rizal

# Data Understanding

Melakukan import library yang dibutuhkan untuk keseluruhan proyek.
"""

# Commented out IPython magic to ensure Python compatibility.
#Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import  OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

"""## Data Loading

Mengunduh dataset yang sudah diupload ke repository github pribadi dan menampilkan isinya.

*Dataset diambil dari https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified*
"""

#Download dataset dari repository github
# !wget -O apartments_for_rent_classified_10K.xlsx https://github.com/Project-Dicoding/Machine_Learning_Terapan_Addin/blob/main/apartments_for_rent_classified_10K.xlsx

#Menampilkan isi dataset
df = pd.read_excel("/content/apartments_for_rent_classified_10K.xlsx")
df

"""Output kode di atas memberikan informasi :


*   Terdapat 10.000 baris
*   Terdapat 22 kolom

## Deskripsi Variabel
Berdasarkan informasi dari sumber dataset berikut adalah penjelasan untuk masing-masing kolom :

* id = unique identifier of apartment
* category = category of classified
* title = title text of apartment
* body = body text of apartment
* amenities = like AC, basketball,cable, gym, internet access, pool, refrigerator etc.
* bathrooms = number of bathrooms
* bedrooms = number of bedrooms
* currency = price in current
* fee = fee
* has_photo = photo of apartment
* pets_allowed = what pets are allowed dogs/cats etc.
* price = rental price of apartment
* price_display = price converted into display for reader
* price_type = price in USD
* square_feet = size of the apartment
* address =  where the apartment is located
* cityname =  where the apartment is located
* state =  where the apartment is located
* latitude = where the apartment is located
* longitude = where the apartment is located
* source = origin of classified
* time = when classified was created

Menampilkan info dari dataset menggunakan .info()
"""

df.info()

"""Output kode di atas memberikan informasi :


*   Terdapat 4 kolom bertipe data float64, 4 kolom bertipe data int64, dan 14 kolom bertipe data object
*   Dengan jumlah data 10.000 terdapat beberapa kolom yang memiliki value Null
*   Kolom price merupakan target

Menampilkan statistik dari dataset menggunakan .describe()
"""

df.describe()

"""Output kode di atas memberikan informasi :

* Count  adalah jumlah sampel pada data.
* Mean adalah nilai rata-rata.
* Std adalah standar deviasi.
* Min yaitu nilai minimum setiap kolom.
* 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% adalah kuartil ketiga.
* Max adalah nilai maksimum.

## Menangani Missing Value dan Outliers

Mencari tahu fitur/kolom mana saja yang memiliki nilai NaN/Null.
"""

# Melakukan cek terhadap kolom mana saja yang memiliki value Null
columns_with_nan = df.columns[df.isnull().any()].tolist()

# Membuat DataFrame untuk menampilkan kolom yang memiliki nilai null beserta tipe datanya
nan_info = pd.DataFrame({
    'Tipe Data': df[columns_with_nan].dtypes,
    'Jumlah Null': df[columns_with_nan].isnull().sum()
})

print("Kolom yang memiliki nilai Null beserta tipe datanya:")
print(nan_info)

"""Output kode di atas memberikan informasi :


*   Terdapat kolom numeric yang memiliki Null Value
*   Terdapat kolom object yang memiliki Null Value
*   Kolom Amenities, pets_allowed, dan address memiliki jumlah missing value yang besar

Cara Mengatasi Missing Values
1.   Drop Missing Values
2.   Imputasi (Pengisian Missing Values)

Kolom dengan jumlah Null kurang dari 100 akan dilakukan drop.

Kolom dengan jumlah Null lebih dari 100 akan dilakukan imputasi.

Menduplikasi dataset supaya dataset original tidak berubah.

Melakukan drop terhadap fitur/kolom yang jumlah Null kurang dari 100.
"""

#Mengcopy dataset agar dataset original tidak terpengaruhi
df_cleaned = df.copy()

# Melakukan drop terhadap kolom yang memiliki jumlah Null kurang dari 100 karena tidak terlalu berpengaruh terhadap jumlah data
df_cleaned = df_cleaned.dropna(subset=['bathrooms', 'bedrooms','cityname','state','latitude','longitude'])

"""Kolom yang memiliki jumlah Null lebih dari 100 adalah amenities, pets_allowed dan address.

Kolom-kolom tersebut merupakan kolom bertipe data objek maka akan dilakukan imputasi terhadap kolom-kolom tersebut dengan nilai "not provided".

Nilai "not provided" dipilih karena belum tentu bahwa tidak memiliki amenities, tidak memperbolehkan hewan, dan tidak memiliki alamat.
"""

# Imputasi untuk kolom kategori
df_cleaned['amenities'].fillna('not provided', inplace=True)
df_cleaned['pets_allowed'].fillna('not provided', inplace=True)
df_cleaned['address'].fillna('not provided', inplace=True)

"""Setelah dilakukan drop dan imputasi dilakukan verifikasi terhadap dataset."""

# Melakukan cek terhadap kolom mana saja yang memiliki value Null
columns_with_nan = df_cleaned.columns[df_cleaned.isnull().any()].tolist()

# Membuat DataFrame untuk menampilkan kolom yang memiliki nilai null beserta tipe datanya
nan_info = pd.DataFrame({
    'Tipe Data': df_cleaned[columns_with_nan].dtypes,
    'Jumlah Null': df_cleaned[columns_with_nan].isnull().sum()
})

print("Kolom yang memiliki nilai Null beserta tipe datanya:")
print(nan_info)
print("Jumlah akhir dataset: ",df_cleaned.shape)

"""Menampilkan kolom-kolom yang bertipe data numerik (int/float)"""

# Menampilkan semua kolom yang bertipe data numerik
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns.tolist()
print("Kolom yang bertipe data numerik:")
print(numeric_columns)

"""Output kode di atas memberikan informasi :
* Terdapat 8 kolom yang bertipe data numerik
* Kolom id, latitude, longtitude, dan time tidak akan dilakukan pengecekan outlier

Kolom 'id', 'latitude', 'longitude', dan 'time' memiliki karakteristik dan kegunaan yang berbeda dari kolom numerik yang biasanya dianalisis untuk outliers. Oleh karena itu, pengecekan outliers pada kolom-kolom ini tidak  diperlukan.

Menggunakan visualisasi boxplot untuk melihat apakah terdapat outliers.
"""

#Kolom numerik yang akan diperiksa outliers nya
numeric_columns = ['bathrooms', 'bedrooms', 'price','square_feet']

# Menampilkan boxplot untuk masing-masing kolom
plt.figure(figsize=(10, 5))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df_cleaned[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

"""Output kode di atas memberikan informasi :
* Beberapa fitur numerik yang dianalisis memiliki outliers

Mengatasi outlier menggunakan metode IQR
"""

#Fungsi untuk menghapus outlier
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#Daftar kolom yang akan diatasi outliersnya
numeric_columns = ['bathrooms', 'bedrooms', 'price','square_feet']

#Menghapus outliers dari masing-masing kolom
for column in numeric_columns:
    df_cleaned = remove_outliers_iqr(df_cleaned, column)

# Verifikasi hasil
df_cleaned.shape

"""Output kode di atas memberikan informasi :
* Dataset sudah bersih
* Total sampel menjadi 8494

## Univariate Analysis

Membagi dataset berdasarkan tipe data yaitu categorical dan numerical.
"""

#Membagi fitur pada dataset menjadi dua bagian
#Fitur yang tidak relevan seperti id, latitude, longtitude, dan time tidak dianalisis
#Fitur memiliki unique value terlalu banyak seperti title, body, amenities, dan address tidak dianalisis
#Fitur yang memiliki value mirip seperti price_display dengan price tidak diambil
#Fitur currency dan fee tidak dianalisis karena memiliki value yang sama untuk keseluruhan dataset

categorical_features = ['category', 'has_photo', 'pets_allowed', 'price_type','cityname', 'state', 'source']
numerical_features = ['bathrooms', 'bedrooms', 'price', 'square_feet']

"""Melakukan analisis terhadap fitur kategori terlebih dahulu."""

#Fungsi untuk menghitung jumlah sampel
#Fungsi untuk menampilkan plot

def analyze_and_plot_category(df, categorical_feature,n=3,m=2):
    """
    Menganalisis dan menampilkan plot untuk fitur kategorikal.

    Parameters:
    df (DataFrame): DataFrame yang mengandung data.
    categorical_feature (str): Nama fitur kategorikal untuk dianalisis.
    """
    # Menganalisis jumlah sampel dan persentase
    count = df[categorical_feature].value_counts()
    percent = 100 * df[categorical_feature].value_counts(normalize=True)
    df_analysis = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})
    print(df_analysis)

    # Menampilkan plot
    plt.figure(figsize=(n, m))
    count.plot(kind='bar', title=categorical_feature)
    plt.xlabel(categorical_feature)
    plt.ylabel('Jumlah')
    plt.show()

"""Melakukan analisis univariate terhadap fitur category."""

#Analisis fitur category
analyze_and_plot_category(df_cleaned, 'category')

"""Output kode di atas memberikan informasi :
* Terdapat 3 kategori yaitu housing/rent/apartement, housing/rent/home, housinig/rent/short_term
* Jumlah kategori housing/rent/home dan housing/rent/short_term sangat sedikit yaitu berjumlah 2 dan 1
* Dapat dilakukan drop terhadap baris yang memiliki value sangat sedikit

Melakukan analisis univariate terhadap fitur has_photo.
"""

#Analisis fitur has_photo
analyze_and_plot_category(df_cleaned, 'has_photo')

"""Output kode di atas memberikan informasi :
* Terdapat 3 kategori yaitu thumbnail, yes, dan no
* Kategori terbanyak adalah thumbnail

Melakukan analisis univariate terhadap fitur pets_allowed.
"""

#Analisis fitur pets_allowed
analyze_and_plot_category(df_cleaned, 'pets_allowed')

"""Output kode di atas memberikan informasi :
* Terdapat 4 kategori yaitu cats/dogs, not provided, cats, dan dogs
* Kategori terbanyak yaitu cats/dogs
* Kategori terbanyak kedua yaitu not provided

Melakukan analisis univariate terhadap fitur price_type.
"""

#Analisis fitur price_type
analyze_and_plot_category(df_cleaned, 'price_type')

"""Output kode di atas memberikan informasi :
* Terdapat dua kategori yaitu monthly dan weekly
* Kategori weekly memiliki jumlah yang sangat sedikit sehingga dapat dilakukan drop

Melakukan analisis univariate terhadap fitur state.
"""

#Analisis fitur state
analyze_and_plot_category(df_cleaned, 'state',8,4)

"""Menghitung jumlah negara bagian (state)."""

#Menghitung jumlah negara bagian (state) yang berbeda
num_unique_states = df['state'].nunique()
print(f"Terdapat {num_unique_states} negara bagian yang berbeda.")

"""Output kode di atas memberikan informasi :
* Terdapat 51 state berbeda
* State dengan jumlah sampel terbanyak adalah TX

Melakukan analisis univariate terhadap fitur source.
"""

#Analisis fitur source
analyze_and_plot_category(df_cleaned, 'source')

"""Output kode di atas memberikan informasi :
* Terdapat 12 kategori berbeda
* Terdapat kategori "False" dengan jumlah 1
* Dapat dilakukan drop terhadap kategori "False"

Melakukan analisis terhadap fitur numerik.

Menampilkan histogram untuk fitur numerik.
"""

#Menampilkan histogram untuk fitur numerik
df_cleaned[numerical_features].hist(bins=50,figsize=(10,5))
plt.show()

"""Output kode di atas memberikan informasi :

1. Bathrooms
 * Mayoritas memiliki 1 kamar mandi. Terdapat rumah yang memiliki 1.5 dan 2.5 kamar mandi
 * Kamar mandi seharusnya tidak berbentuk float sehingga akan dilakukan drop
2. Bedrooms
 * Mayoritas memiliki 1 atau 2 kamar tidur.
3. Price
 * Harga sewa apartemen tersebar dengan puncak sekitar 700-1000 USD. Distribusi harga menunjukkan pola normal dengan sedikit skew ke kanan.
4. Square_feet
 * Luas apartemen bervariasi dengan puncak sekitar 500-800 kaki persegi. Distribusi luas juga menunjukkan pola normal dengan skew ke kanan.

Berdasarkan analisis univariate ditemukan beberapa sampel yang akan dilakukan drop.

Menampilkan jumlah baris dan kolom dari dataset sebelum dilakukan penghapusan.
"""

df_cleaned.shape

"""Menghapus sampel-sampel berdasarkan hasil analisis uni variates."""

#Menghapus data yang value pada kolom category bernilai "housing/rent/home" dan "housing/rent/short_term"
df_cleaned = df_cleaned[~df_cleaned['category'].isin(['housing/rent/home', 'housing/rent/short_term'])]

#Menghapus data yang value pada kolom price_type bernilai "weekly"
df_cleaned = df_cleaned[df_cleaned['price_type'] != 'Weekly']

#Menghapus data yang value pada kolom source bernilai "False"
df_cleaned = df_cleaned[df_cleaned['source'] != 'FALSE']

#Menghapus data yang memiliki nilai non-bulat pada kolom bathrooms
df_cleaned = df_cleaned[df_cleaned['bathrooms'] % 1 == 0]

"""Menampilkan jumlah baris dan kolom dari dataset setelah dilakukan penghapusan."""

df_cleaned.shape

"""Output kode di atas memberikan informasi :
* Setelah dilakukan cleaning lebih lanjut total data yang dimiliki menjadi 8136

Melakukan verifikasi kolom/fitur mana saja yang memiliki satu nilai untuk untuk keseluruhan dataset setelah dilakukan penghapusan.
"""

#Menampilkan kolom yang hanya memiliki satu nilai unik
single_value_columns = [col for col in df_cleaned.columns if df_cleaned[col].nunique() == 1]

print("Kolom yang hanya memiliki satu nilai unik untuk keseluruhan dataset:")
print(single_value_columns)

"""## Multivariate Analysis

Categorical Features.

Menampilkan plot categorical fitur terhadap price.
"""

#Daftar fitur kategorikal dalam dataset
#Fitur yang tidak memiliki pengaruh dan jumlah unik value yang terlalu besar tidak dianalisis
categorical_features = ['has_photo','pets_allowed','state','source']

#Membuat plot untuk setiap fitur kategorikal
for col in categorical_features:
    sns.catplot(x=col, y="price", kind="bar", dodge=False, height=4, aspect=3, data=df_cleaned, palette="Set3")
    plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))
    plt.xticks(rotation=90)
    plt.show()

"""Output kode di atas memberikan informasi :
* Semua fitur hampir memberikan pengaruh yang cenderung mirip terhadap harga

Numerical Features.

Menampilkan plot hubungan antar fitur numerik.
"""

#Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
numerical_features = ['bathrooms','bedrooms','price','square_feet']
sns.pairplot(df_cleaned[numerical_features], diag_kind = 'kde')

"""Menampilkan heatmap untuk korelasi antar fitur numerik."""

#Menampilkan heatmap untuk korelasi antar fitur numerik
plt.figure(figsize=(10, 8))
correlation_matrix = df_cleaned[numerical_features].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Output kode di atas memberikan informasi :
1. Hubungan antara Luas Bangunan dan Fitur Lain:
 * Fitur square_feet (luas bangunan) memiliki hubungan yang cukup kuat dengan bathrooms dan bedrooms, yang menunjukkan bahwa luas bangunan adalah indikator penting dari ukuran dan fasilitas apartemen.
2. Harga Sewa:
 * Korelasi antara harga sewa (price) dengan fitur lain (kamar mandi dan kamar tidur) relatif rendah. Ini menunjukkan bahwa harga sewa mungkin dipengaruhi oleh faktor-faktor lain seperti lokasi, fasilitas tambahan, atau kondisi pasar yang tidak tercakup dalam fitur yang dianalisis.

# Data Preparation

Pada bagian ini kita akan melakukan empat tahap persiapan data, yaitu:

* Drop fitur yang tidak memberikan nilai tambahan.
* Encoding fitur kategori.
* Pembagian dataset dengan fungsi train_test_split dari library sklearn.
* Standarisasi.

Fitur-fitur ini di-drop karena tidak memberikan nilai tambah untuk analisis atau model prediksi:


1. id (Tidak memberikan nilai tambah)
2. category (Nilainya sama untuk keseluruhan dataset)
3. title (Tidak memberikan nilai tambah)
4. body (Tidak memberikan nilai tambah)
5. currency (Nilainya sama untuk keseluruhan dataset)
6. fee (Nilainya sama untuk keseluruhan dataset)
7. price_display (Memiliki nilai yang sama dengan fitur `price`)
8. price_type (Nilainya sama untuk keseluruhan dataset)
9. address (Tidak memberikan nilai tambah)
10. latitude (Tidak memberikan nilai tambah)
11. longitude (Tidak memberikan nilai tambah)
12. time (Tidak memberikan nilai tambah)
"""

#Melakukan drop kolom
features_to_drop = ['id', 'category','title', 'body','currency', 'fee','price_display','price_type','address','latitude','longitude','time']
df_cleaned = df_cleaned.drop(columns=features_to_drop)

"""Menampilkan dataset setelah dilakukan drop fitur yang tidak memberikan nilai tambah."""

#Menampilkan dataset setelah dilakukan drop
df_cleaned.head()

"""Output kode di atas memberikan informasi :
* Jumlah fitur tersisa 10 dari 22

Menerapkan one-hot-encoding terhadap fitur kategorikal.
"""

#Mengubah fitur kategorikal menjadi numerik
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['amenities'], prefix='amenities')], axis=1)
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['has_photo'], prefix='has_photo')], axis=1)
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['pets_allowed'], prefix='pets_allowed')], axis=1)
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['cityname'], prefix='cityname')], axis=1)
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['state'], prefix='state')], axis=1)
df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned['source'], prefix='source')], axis=1)

#Menghapus kolom kategorikal asli
df_cleaned.drop(['amenities', 'has_photo', 'pets_allowed','cityname', 'state', 'source'], axis=1, inplace=True)

#Menampilkan DataFrame yang telah dibersihkan
df_cleaned.head()

"""Output kode di atas memberikan informasi :
* Telah berhasil dilakukan one-hot-encoding namun value masih berupa True dan False
* Harus dilakukan pengubahan value menjadi numerik

Mengubah nilai True dan False menjadi 1 dan 0.
"""

#Mengubah nilai True dan False menjadi 1 dan 0
df_cleaned = df_cleaned.astype(int)
df_cleaned.head(1)

"""Train-Test-Split

Membagi dataset menjadi fitur dan target. Target merupakan fitur price. Pembagian dataset dengan perbandingan 80:20.
"""

#Kolom yang menjadi target adalah kolom price
X = df_cleaned.drop(["price"],axis =1)
y = df_cleaned["price"]

#Pembagian data adalah 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

"""Menampilkan jumlah sampel pada masing-masing bagian."""

#Menampilkan jumlah sampel pada masing-masing bagian
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Standarisasi

Melakukan standarirasi agar performa dari algoritma machine learningi dapat berjalan lebih optimal.
"""

#Melakukan standarisasi terhadap fitur numerik pada data train
numerical_features = ['bathrooms','bedrooms','square_feet']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""Melakukan verifikasi bahwa standarisasi berhasil dengan melihata nilai mean = 0 dan std = 1."""

#Melakukan cek terhadap nilai mean dan std
X_train[numerical_features].describe().round(4)

"""# Modeling

Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma.
1. K-Nearest Neighbor
2. Random Forest
3. Boosting Algorithm

Menyiapkan dataframe untuk menyimpan hasil analisis masing-masing model.
"""

#Menyiapkan dataframe untuk analisis masing-masing model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""Membuat model KNN dan menyimpan hasilnya."""

#Melatih model KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""Membuat model RF dan menyimpan hasilnya."""

#Melatih model Random Forest
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""Membuat model AdaBoost dan menyimpan hasilnya."""

#Melatih model Boosting Algorithm
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""# Evaluation

Melakukan standarisasi terhadap fitur numerical untuk data test.
"""

#Melakukan standarisasi terhadap fitur numerik pada data test
numerical_features = ['bathrooms','bedrooms','square_feet']
scaler = StandardScaler()
scaler.fit(X_test[numerical_features])
X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_test[numerical_features].head()

"""Menampilkan nilai MSE untuk masing-masing model."""

#Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

#Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

#Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

#Print mse
mse

"""Membuat visualisasi agar perbandingan nilai MSE menjadi lebih mudah."""

#Plot metrik dengan bar chart
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""Output kode di atas memberikan informasi :
* Model random forest memberikan nilai error yang paling kecil sedangkan model boosting memberikan nilai error yang paling besar.

Membuat visualisasi menggunakan model yang memberikan nilai MSE paling kecil yaitu RF.
"""

#Hitung MSE untuk data latih dan data uji
train_mse = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
test_mse = mean_squared_error(y_pred=RF.predict(X_test), y_true=y_test)

# impan nilai MSE dalam DataFrame untuk visualisasi
models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['RandomForest'])
models.loc['train_mse', 'RandomForest'] = train_mse
models.loc['test_mse', 'RandomForest'] = test_mse

#Buat prediksi pada data uji
y_pred = RF.predict(X_test)

#Visualisasi hasil prediksi vs nilai aktual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices for RandomForest')
plt.show()