import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pickle

# Load your dataset
data = pd.read_csv('your_data.csv')

# Check class distribution
print(data['label'].value_counts())

# Separate the classes
gastric = data[data['label'] == 'gastric']
migraine = data[data['label'] == 'migraine']

# Downsample the majority class or upsample the minority class
if len(gastric) > len(migraine):
    gastric_downsampled = resample(gastric, 
                                   replace=False,  # Without replacement
                                   n_samples=len(migraine),  # Match minority class
                                   random_state=42)  # Reproducible results
    balanced_data = pd.concat([gastric_downsampled, migraine])
else:
    migraine_upsampled = resample(migraine, 
                                  replace=True,  # With replacement
                                  n_samples=len(gastric),  # Match majority class
                                  random_state=42)  # Reproducible results
    balanced_data = pd.concat([gastric, migraine_upsampled])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset (optional)
balanced_data.to_csv('balanced_data.csv', index=False)

# Split features and labels
X = balanced_data[['Vata_pressure', 'Pitta_pressure', 'Kapha_pressure', 'Oxygen_level']]
y = balanced_data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model to a file
with open('naadi_diagnostic_model_SMOT.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
