import pandas as pd
from sklearn.utils import resample

# Sample data creation
data = {
    'Vata_pressure': [10, 20, 30, 40, 50, 60, 10, 20, 30, 40],
    'Pitta_pressure': [15, 25, 35, 45, 55, 65, 15, 25, 35, 45],
    'Kapha_pressure': [12, 22, 32, 42, 52, 62, 12, 22, 32, 42],
    'Oxygen_level': [95, 96, 97, 98, 99, 100, 95, 96, 97, 98],
    'label': ['gastric', 'gastric', 'gastric', 'gastric', 'gastric', 'gastric',
              'migraine', 'migraine', 'migraine', 'migraine']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separate the classes
gastric = df[df['label'] == 'gastric']
migraine = df[df['label'] == 'migraine']

# Downsample the majority class (gastric) to match the minority class (migraine)
gastric_downsampled = resample(gastric, 
                               replace=False,  # Without replacement
                               n_samples=len(migraine),  # Match minority class
                               random_state=42)  # Reproducible results

# Create balanced dataset
balanced_data = pd.concat([gastric_downsampled, migraine])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_data.to_csv('balanced_data.csv', index=False)

print("Balanced dataset created and saved as 'balanced_data.csv'!")
