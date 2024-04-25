from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('data/soil_data.csv')

# Assuming your dataset has features related to soil properties and target as crop type
X = data.drop('Crop_Type', axis=1)  # Features
y = data['Crop_Type']  # Target variable

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        soil_data = [float(request.form['ph']), float(request.form['humidity']), float(request.form['temperature']), float(request.form['nitrogen']), float(request.form['phosphorus']), float(request.form['potassium'])]

        # Predict crop type probabilities
        probabilities = clf.predict_proba([soil_data])[0]

        # Get the top 3 predicted crops
        top_crops_indices = probabilities.argsort()[-3:][::-1]
        top_crops = [clf.classes_[i] for i in top_crops_indices]
        top_probabilities = [probabilities[i] for i in top_crops_indices]

        return render_template('result.html', crops_probabilities=zip(top_crops, top_probabilities))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
