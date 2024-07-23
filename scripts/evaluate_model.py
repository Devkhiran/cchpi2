import joblib
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the model and preprocessed data
model = load_model('models/lstm_model.h5')
_, _, _, _, tokenizer, label_encoder, max_sequence_length, _, _ = joblib.load('data/preprocessed_data.pkl')

def get_top_n_predictions(model, tokenizer, label_encoder, new_cchpi, n=5):
    # Clean and tokenize the new CCHPI
    new_cchpi_clean = new_cchpi.lower().replace(r'[^\w\s]', '')
    new_sequence = tokenizer.texts_to_sequences([new_cchpi_clean])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)

    # Get probability predictions
    probabilities = model.predict(new_padded_sequence)[0]

    # Get top N predictions
    top_n_indices = np.argsort(probabilities)[-n:][::-1]
    top_n_icd_names = label_encoder.inverse_transform(top_n_indices)
    top_n_probabilities = probabilities[top_n_indices]

    return list(zip(top_n_icd_names, top_n_probabilities))

# Example usage
new_cchpi = "patient complains of severe headache and dizziness"
top_5_predictions = get_top_n_predictions(model, tokenizer, label_encoder, new_cchpi, n=5)
print("Top 5 predictions:")
for icd_name, probability in top_5_predictions:
    print(f"ICD Name: {icd_name}, Probability: {probability:.4f}")