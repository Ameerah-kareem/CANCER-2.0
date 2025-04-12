
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# Load the dataset
df = pd.read_csv("Downloads/Breast cancer dataset cleaned.csv")

# Prepare features and target
X = df.drop(columns=['Diagnosis'])
Y = df['Diagnosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0015), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.15, verbose=0, callbacks=[early_stopping])

# Prediction for test set
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

# Streamlit App
logo_path = r"Downloads/logo.png"
st.set_page_config(layout="wide")
st.image(logo_path, use_column_width='auto')

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "About"],
    icons=["house", "table", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Home":
    st.title("BreastGuard: Machine Learning-Based Breast Cancer Prediction System")
    st.write("Please Input Corresponding Patient Data")

    st.header('Patient Input Parameters')

    def user_input_features():
        Age = st.number_input('Enter Patient Age', min_value=1, max_value=120, step=1)
        gender_options = {"Male": 0, "Female": 1}
        Gender = st.selectbox('Gender', list(gender_options.keys()))
        Gender = gender_options[Gender]

        laterality_options = {"Left": 1, "Right": 0}
        laterality = st.selectbox('Laterality', list(laterality_options.keys()))
        laterality = laterality_options[laterality]

        lymph_node_options = {"No": 0, "Yes": 1}
        Lymph_Node = st.selectbox('Lymph Node', list(lymph_node_options.keys()))
        Lymph_Node = lymph_node_options[Lymph_Node]

        Nature_of_Aspirate = st.selectbox(
            'Nature of Aspirate',
            ["colloid_Aspirate", "creamy_Aspirate", "hemorrhagic_Aspirate",
            "milky_Aspirate", "mucoid_Aspirate", "oily_Aspirate",
            "proteinaceous_Aspirate", "sanguineous_Aspirate",
            "serous_Aspirate", "turbid_Aspirate"]
        )

        data = {
            'Age': Age,
            'Gender': Gender,
            'laterality': laterality,
            'Lymph_Node': Lymph_Node,
            'Nature_of_Aspirate': Nature_of_Aspirate
        }

        features = pd.DataFrame(data, index=[0])
        features = features.reindex(columns=X_train.columns, fill_value=0)
        return features

    input_df = user_input_features()

    if st.button('Submit'):
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob_malignant = prediction[0][0]
        prob_benign = 1 - prob_malignant

        st.subheader('Prediction')

        if prob_malignant < 0.70 and prob_malignant > 0.5:
            st.write("Most likely malignant, further testing advised.")
        elif prob_malignant > 0.70:
            st.write(f"Predicted Diagnosis: {'Malignant' if prob_malignant >= 0.5 else 'Benign'}")

        if prob_benign < 0.70 and prob_benign > 0.5:
            st.write("Most likely benign, further testing advised.")
        elif prob_benign > 0.70:
            st.write(f"Predicted Diagnosis: {'Benign' if prob_malignant < 0.5 else 'Malignant'}")

        st.subheader('Prediction Probability')
        st.write(f"Probability of Malignant: {prob_malignant:.2f}")
        st.write(f"Probability of Benign: {prob_benign:.2f}")

elif selected == "Dataset":
    st.title("Dataset Preview")
    st.write("Below is a preview of the breast cancer dataset used:")
    st.dataframe(df)

elif selected == "About":
    st.title("About BreastGuard")
    st.write("""
    BreastGuard is an intelligent diagnostic support system that uses deep learning
    to assist medical professionals in predicting breast cancer from aspirate characteristics
    and patient details. This application is built with TensorFlow and Streamlit for real-time
    interaction and intuitive results visualization.
    """)
