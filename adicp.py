import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
df = pd.read_csv('ADIP-CI_Postoperative_Rehabilitation_Data_Sample.csv')

# Preprocess the data
df = df.dropna()
df = pd.get_dummies(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Rehabilitation_Outcome', axis=1), df['Rehabilitation_Outcome'], test_size=0.25, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
model.evaluate(X_test, y_test)

# Make predictions on new data
new_data = np.array([[1, 2, 3, 4]])
y_pred = model.predict(new_data)

# Print the predicted rehabilitation outcome
print('Predicted rehabilitation outcome:', y_pred[0])