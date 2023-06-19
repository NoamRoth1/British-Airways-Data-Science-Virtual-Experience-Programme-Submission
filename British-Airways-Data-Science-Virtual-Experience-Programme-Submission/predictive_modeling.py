import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/customer_booking.csv", encoding="ISO-8859-1")

mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df['flight_day'] = df['flight_day'].map(mapping)

# Perform one-hot encoding on the sales_channel column
encoded_column = pd.get_dummies(df['sales_channel'], dtype=int)

# Concatenate the encoded column with the original DataFrame
df = pd.concat([df, encoded_column], axis=1)

# Drop the original sales_channel column
df = df.drop('sales_channel', axis=1)

# Perform one-hot encoding on the trip_type column
encoded_column = pd.get_dummies(df['trip_type'], dtype=int)

# Concatenate the encoded column with the original DataFrame
df = pd.concat([df, encoded_column], axis=1)

# Drop the original sales_channel column
df = df.drop('trip_type', axis=1)

print(df.info())

le = LabelEncoder()
df['booking_origin'] = le.fit_transform(df['booking_origin'])
df['route'] = le.fit_transform(df['route'])

print(df.info())

# Save the new Data Frame to customer_booking_encoded.csv
df.to_csv('customer_booking_encoded.csv', index=False)

# Read the data from customer_booking_encoded.csv
df = pd.read_csv('customer_booking_encoded.csv')

# Prepare the data
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the RandomForestClassifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get feature importances
importances = model.feature_importances_

# Get feature names
feature_names = X.columns

# Sort feature importances in descending order
indices = importances.argsort()[::-1]

# Rearrange feature names based on feature importances
sorted_feature_names = [feature_names[i] for i in indices]

# Convert importances to percentages
importances_percent = importances[indices] * 100

# Set a threshold for feature importance
threshold = 1.0  # Adjust the threshold as needed

# Filter features above the threshold
filtered_feature_names = [name for name, importance in zip(sorted_feature_names, importances_percent) if importance >= threshold]
filtered_importances_percent = [importance for importance in importances_percent if importance >= threshold]

# Calculate the total importance of variables below the threshold
below_threshold_importance = sum(importance for importance in importances_percent if importance < threshold)

# Group the variables below the threshold as "Other"
other_feature_names = [name for name, importance in zip(sorted_feature_names, importances_percent) if importance < threshold]
other_importances_percent = [importance for importance in importances_percent if importance < threshold]
filtered_feature_names.append("Other")
filtered_importances_percent.append(below_threshold_importance)

# Plot feature importances as a pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(filtered_importances_percent, labels=filtered_feature_names, autopct='%1.1f%%')
ax.set_title('Feature Importance')

# Create a small box in the bottom left corner for "Other" variables
bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="gray", lw=0.5)
other_text = "Other:\n"
for i, (name, importance) in enumerate(zip(other_feature_names, other_importances_percent)):
    if i == len(other_feature_names) - 1:
        other_text += f"{name}: {importance:.1f}%"
    else:
        other_text += f"{name}: {importance:.1f}%\n"
ax.text(0.02, 0.02, other_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=bbox_props)

plt.tight_layout()

plt.savefig('BA Feature Importance pie chart.png')