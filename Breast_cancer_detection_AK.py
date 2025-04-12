
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# Load dataset
df = pd.read_csv("Breast_cancer_dataset_refined_2_0.csv")

# Prepare features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0015), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.15, verbose=0, callbacks=[early_stopping])

# Evaluate on test
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

# UI
logo_path = "images/image.jpeg"
st.image(logo_path, use_container_width=True)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "About"],
    icons=["house", "table", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# ---------------- Home / Prediction Page ----------------
if selected == "Home":
    st.title("Ameerah's SmartPredict: A Machine Learning-Based Breast Cancer Prediction System")
    st.markdown("Please input patient information to predict the likelihood of breast cancer.")

    st.markdown("### 👩‍⚕️ Patient Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input('Patient Age', min_value=1, max_value=120)
        Gender = st.selectbox('Gender', ['Male', 'Female'])
        familial_cancer = st.selectbox('Family History of Cancer', ['No', 'Yes'])

    with col2:
        Laterality = st.selectbox('Laterality', ['Left', 'Right'])
        Lymph_Node = st.selectbox('Lymph Node', ['No', 'Yes'])
        Tumor_shape = st.selectbox('Tumor Shape', 
            ['lobulated_shape', 'nodular_shape', 'oval_shape', 'round_shape', 'stellate_shape'])

    Nature_of_Aspirate = st.selectbox(
        'Nature of Aspirate',
        ["colloid_Aspirate", "creamy_Aspirate", "hemorrhagic_Aspirate",
        "milky_Aspirate", "mucoid_Aspirate", "oily_Aspirate",
        "proteinaceous_Aspirate", "sanguineous_Aspirate",
        "serous_Aspirate", "turbid_Aspirate"]
    )

    gender_map = {"Male": 0, "Female": 1}
    laterality_map = {"Left": 1, "Right": 0}
    lymph_node_map = {"No": 0, "Yes": 1}
    fam_cancer_map = {"No": 0, "Yes": 1}
        
    data = {
        'Age': Age,
        'Gender': gender_map[Gender],
        'laterality': laterality_map[Laterality],
        'Lymph_Node': lymph_node_map[Lymph_Node],
        'Nature_of_Aspirate': Nature_of_Aspirate,
        'familial_cancer': fam_cancer_map[familial_cancer],
        'Tumor_shape': Tumor_shape
    }

    input_df = pd.DataFrame(data, index=[0])
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    if st.button('Submit'):
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob_malignant = prediction[0][0]
        prob_benign = 1 - prob_malignant

        st.subheader("🧬 Prediction Result")

        if prob_malignant > 0.7:
            st.error("🔴 High Risk of Malignancy — Please seek further testing.")
        elif prob_malignant > 0.5:
            st.warning("🟠 Moderate Risk — Clinical follow-up advised.")
        elif prob_benign > 0.7:
            st.success("🟢 Low Risk — Likely Benign.")
        else:
            st.info("🧭 Uncertain — Retesting may be necessary.")

        st.markdown("### 📊 Prediction Probability")
        st.metric("Malignant Probability", f"{prob_malignant:.2%}")
        st.metric("Benign Probability", f"{prob_benign:.2%}")


# ---------------- Dataset Page ----------------
elif selected == "Dataset":
    st.title("📊 Dataset Preview")
    st.write("Below is the dataset used to train the prediction model.")
    st.dataframe(df)

    st.markdown("### 🧮 Class Distribution")
    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.countplot(x='Diagnosis', data=df, palette='pastel', ax=ax)
    st.pyplot(fig)

# ---------------- About Page ----------------
elif selected == "About":
    st.title("ℹ️ About Ameerah's SmartPredict")
    st.markdown("""
    **Ameerah's SmartPredict** is an intelligent diagnostic support tool built to enhance the early detection of breast cancer using advanced machine learning techniques.

This application is specifically designed to support laboratory technicians in Nigeria, where there remains a significant gap in the integration of modern technology within diagnostic workflows.
By leveraging patient data and aspirate characteristics, SmartPredict offers quick, reliable predictions to aid clinical decision-making — even in resource-limited settings.
    
     
    - Developed by Ameerah Kareem  
  Bridging the gap between traditional lab practices and smart diagnostic tools

    """)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center style='color: gray;'>Made with 💙 by Ameerah | Powered by Streamlit + TensorFlow</center>", unsafe_allow_html=True)
