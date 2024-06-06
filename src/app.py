from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the pickled SVM model
with open('bcmodel9.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        mean_radius = float(request.form['mean_radius'])
        mean_texture = float(request.form['mean_texture'])
        mean_perimeter = float(request.form['mean_perimeter'])
        mean_area = float(request.form['mean_area'])
        mean_smoothness = float(request.form['mean_smoothness'])
        mean_compactness = float(request.form['mean_compactness'])
        mean_concavity = float(request.form['mean_concavity'])
        mean_concavepoints = float(request.form['mean_concavepoints'])
        mean_symmetry = float(request.form['mean_symmetry'])

        # Create a numpy array with the input data
        input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,mean_compactness,mean_concavity,mean_concavepoints,mean_symmetry]])
        model = joblib.load('bcmodel9.pkl')
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)

        # Convert the prediction to a human-readable result
        result = 'MALIGNANT' if prediction[0] == 1 else 'BENIGN'
        print(result)
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction='Error: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
