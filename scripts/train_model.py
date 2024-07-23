from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import joblib
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd


# Load the preprocessed data
X_train, X_test, y_train, y_test, tokenizer, label_encoder, max_sequence_length, embedding_matrix, num_classes = joblib.load('data/preprocessed_data.pkl')

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Build the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_matrix.shape[1], 
                    weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=32)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f%%' % (accuracy * 100))

# Save the model
model.save('models/lstm_model.h5')

print("Model training complete and model saved.")








# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)

# Convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# Save the report
df_report.to_csv('classification_report.csv')

# Visualize the results
plt.figure(figsize=(20,10))
sns.heatmap(df_report[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues')
plt.title('Classification Metrics for Each ICD Name')
plt.tight_layout()
plt.savefig('classification_metrics.png')
plt.close()

# Print overall accuracy
print(f"Overall Accuracy: {report['accuracy']:.2f}")