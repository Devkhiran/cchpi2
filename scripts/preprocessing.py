import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from gensim.models import Word2Vec # type: ignore
import joblib
import numpy as np

# Load the data
data_path = 'data/dev 2 (Autosaved).xlsx'
data = pd.read_excel(data_path)

# Clean CCHPI
data['cchpi_clean'] = data['cchpi'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['cchpi_clean'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(data['cchpi_clean'])
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
joblib.dump(tokenizer, 'data/tokenizer.pkl')
joblib.dump(max_sequence_length, 'data/max_sequence_length.pkl')
# Word2Vec
embedding_dim = 100
sentences = [sentence.split() for sentence in data['cchpi_clean']]
word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Create embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
joblib.dump(embedding_matrix, 'data/embedding_matrix.pkl')
# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['icdname'])
num_classes = len(data['icdname'].unique())
joblib.dump(label_encoder, 'data/label_encoder.pkl')
# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Save the preprocessed data
joblib.dump((X_train, X_test, y_train, y_test, tokenizer, label_encoder, max_sequence_length, embedding_matrix, num_classes), 'data/preprocessed_data.pkl')

print("Preprocessing complete and data saved.")