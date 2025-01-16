import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Title and Description
st.title('Analisis Data Balita')
st.write('Aplikasi ini menganalisis data balita dan memprediksi status gizi menggunakan Logistic Regression.')

# File Uploader
st.header('Upload Dataset')
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    
    # Show Dataset Information
    st.header('Informasi Dataset')
    st.write(df.info())
    st.write(df.head())
    st.write(df.describe())

    # Visualizations
    st.header('Distribusi Umur Balita')
    sns.histplot(df['Umur (bulan)'])
    st.pyplot(plt.gcf())

    st.header('Distribusi Status Gizi')
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['Status Gizi'], order=df['Status Gizi'].value_counts().index)
    plt.title('Distribusi Jumlah Stunting')
    plt.xlabel('Jumlah')
    plt.ylabel('Kategori Status Gizi')
    st.pyplot(plt.gcf())

    # Handle Missing Values
    st.header('Missing Values')
    missing_values = df.isnull().sum()
    st.write(missing_values)

    df.fillna(method='ffill', inplace=True)

    # Prepare Data for Model
    X = df[['Umur (bulan)', 'Jenis Kelamin']]
    X = pd.get_dummies(X, columns=['Jenis Kelamin'], drop_first=True)
    y = df['Status Gizi']

    # Model Parameters Input
    st.header('Pengaturan Model')
    test_size = st.slider('Ukuran Data Uji', min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    random_state = st.number_input('Random State', min_value=0, value=42, step=1)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.header('Classification Report')
    st.text(classification_report(y_test, y_pred))

    st.header('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.pyplot(plt.gcf())
else:
    st.write("Silakan unggah file CSV untuk memulai analisis.")
