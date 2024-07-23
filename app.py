import streamlit as st
import pandas as pd
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Attempt to load the model and preprocessed data
try:
    model = load_model('models/lstm_model.h5')
    _, _, _, _, tokenizer, label_encoder, max_sequence_length, _, _ = joblib.load('data/preprocessed_data.pkl')
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model or preprocessed data: {str(e)}")
    st.error("The app will continue with limited functionality.")
    model_loaded = False

# Load Excel data
try:
    drug_data = pd.read_excel("data/ICD_drug2_data.xlsx")
    investigation_data = pd.read_excel("data/ICD_investigation_data.xlsx")
except Exception as e:
    st.error(f"Error loading Excel data: {str(e)}")
    st.error("Please make sure the Excel files are in the correct location.")
    drug_data = pd.DataFrame()
    investigation_data = pd.DataFrame()

def get_top_n_predictions(new_cchpi, n=5):
    if not model_loaded:
        return ["Model not loaded"] * n
    
    new_cchpi_clean = new_cchpi.lower().replace(r'[^\w\s]', '')
    new_sequence = tokenizer.texts_to_sequences([new_cchpi_clean])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)
    
    probabilities = model.predict(new_padded_sequence)[0]
    top_n_indices = np.argsort(probabilities)[-n:][::-1]
    top_n_icd_names = label_encoder.inverse_transform(top_n_indices)
    
    return list(top_n_icd_names)

def main():
    st.title("Medical Recommendation System")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 0

    # Navigation
    if st.session_state.page == 0:
        patient_info_page()
    elif st.session_state.page == 1:
        drug_selection_page()
    elif st.session_state.page == 2:
        investigation_selection_page()

def patient_info_page():
    st.header("Patient Information")
    
    st.session_state.name = st.text_input("Patient Name")
    st.session_state.age = st.number_input("Age", min_value=0, max_value=120)
    st.session_state.gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.session_state.diabetes = st.checkbox("Diabetes")
    st.session_state.hypertension = st.checkbox("Hypertension")
    st.session_state.other_illness = st.checkbox("Other Illness")
    
    if st.session_state.other_illness:
        st.session_state.other_illness_description = st.text_input("Please specify other illness")
    
    st.session_state.cchpi = st.text_area("Chief Complaint and History of Present Illness (CCHPI)")
    
    if st.session_state.cchpi:
        top_5_icds = get_top_n_predictions(st.session_state.cchpi)
        
        # Option to switch between model suggestions and full list
        use_model_suggestions = st.checkbox("Use model suggestions for ICD", value=True)
        
        if use_model_suggestions:
            st.session_state.selected_icd = st.selectbox("Select ICD (Top 5 Suggestions)", top_5_icds)
        else:
            # Full list of ICDs
            all_icds = [
                "1 Vessel CAD", "2 Vessel CAD", "3 Vessel CAD", "Acute Coronary Syndrome",
                "Acute Heart Failure Syndrome", "Acute Stroke", "Atherosclerotic Heart Disease",
                "Chronic Kidney Disease", "Diabetes Mellitus", "Dyslipidemia",
                "Essential (Primary) Hypertension", "Fracture Lower Limb", "Gastritis and Duodenitis",
                "Non-Insulin-Dependent Diabetes Mellitus Type 2 without Complications",
                "Non-ST Elevation Myocardial Infarction (NSTEMI)", "Old Myocardial Infarction",
                "Osteoarthritis", "Peripheral Arterial Disease",
                "ST Elevation Myocardial Infarction (STEMI)", "Systemic Hypertension",
                "Unspecified Diabetes Mellitus without Complications"
            ]
            st.session_state.selected_icd = st.selectbox("Select ICD from full list", all_icds)
    
    if st.button("Proceed to Drug Selection"):
        st.session_state.page = 1

def drug_selection_page():
    st.header("Drug Selection")
    
    # Filter drugs based on selected ICD
    relevant_drugs = drug_data[drug_data['ICDname'] == st.session_state.selected_icd]['Drug'].tolist()
    
    if 'selected_drugs' not in st.session_state:
        st.session_state.selected_drugs = []
    
    # Add new drug selections
    new_drug = st.selectbox("Select Drug", [""] + relevant_drugs)
    if new_drug:
        remarks = st.text_input("Remarks for " + new_drug)
        if st.button("Add Drug"):
            st.session_state.selected_drugs.append((new_drug, remarks))
    
    # Display selected drugs
    if st.session_state.selected_drugs:
        st.write("Selected Drugs:")
        for i, (drug, remarks) in enumerate(st.session_state.selected_drugs):
            st.write(f"{i+1}. {drug}: {remarks}")
    
    if st.button("Proceed to Investigations"):
        st.session_state.page = 2

def investigation_selection_page():
    st.header("Investigation Selection")
    
    # Filter investigations based on selected ICD
    relevant_investigations = investigation_data[investigation_data['ICDname'] == st.session_state.selected_icd]['Investigation'].tolist()
    
    if 'selected_investigations' not in st.session_state:
        st.session_state.selected_investigations = []
    
    # Add new investigation selections
    new_investigation = st.selectbox("Select Investigation", [""] + relevant_investigations)
    if new_investigation and st.button("Add Investigation"):
        st.session_state.selected_investigations.append(new_investigation)
    
    # Display selected investigations
    if st.session_state.selected_investigations:
        st.write("Selected Investigations:")
        for i, investigation in enumerate(st.session_state.selected_investigations):
            st.write(f"{i+1}. {investigation}")
    
    if st.button("Generate PDF"):
        pdf_file = create_pdf_report()
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="medical_report.pdf",
            mime="application/pdf"
        )

def create_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph(f"Patient: {st.session_state.name}", styles['Heading1']))
    elements.append(Paragraph(f"Age: {st.session_state.age}", styles['Normal']))
    elements.append(Paragraph(f"Gender: {st.session_state.gender}", styles['Normal']))
    
    conditions = []
    if st.session_state.diabetes:
        conditions.append("Diabetes")
    if st.session_state.hypertension:
        conditions.append("Hypertension")
    if st.session_state.other_illness:
        conditions.append(st.session_state.other_illness_description)
    
    if conditions:
        elements.append(Paragraph("Medical Conditions:", styles['Heading2']))
        for condition in conditions:
            elements.append(Paragraph(f"- {condition}", styles['Normal']))
    
    elements.append(Paragraph(f"ICD: {st.session_state.selected_icd}", styles['Heading2']))
    
    elements.append(Paragraph("Prescribed Drugs:", styles['Heading2']))
    if st.session_state.selected_drugs:
        for drug, remarks in st.session_state.selected_drugs:
            elements.append(Paragraph(f"- {drug}: {remarks}", styles['Normal']))
    else:
        elements.append(Paragraph("No drugs prescribed.", styles['Normal']))
    
    elements.append(Paragraph("Recommended Investigations:", styles['Heading2']))
    if st.session_state.selected_investigations:
        for investigation in st.session_state.selected_investigations:
            elements.append(Paragraph(f"- {investigation}", styles['Normal']))
    else:
        elements.append(Paragraph("No investigations recommended.", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()