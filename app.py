from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import requests
import pprint
app = Flask(__name__)

# # Load your model
# model = pickle.load(open('./crop.pkl', 'rb'))

model_paths = {
    "KNeighborsClassifier" : "models/KNeighborsClassifier.pkl",
    "SVC" : "models/SVC.pkl",
    "DecisionTreeClassifier" : "models/DecisionTreeClassifier.pkl",
    "LogisticRegression" : "models/LogisticRegression.pkl",
    "GaussianNB" : "models/GaussianNB.pkl",
    "GradientBoostingClassifier" : "models/GradientBoostingClassifier.pkl",
    "RandomForestClassifier" : "models/RandomForestClassifier.pkl",
    "knn_model" : "models/knn_model.pkl",
    "svm_model" : "models/svm_model.pkl",
    "decision_tree_model" : "models/decision_tree_model.pkl",
    "logistic_regression_model" : "models/logistic_regression_model.pkl",
    "naive_bayes_model" : "models/naive_bayes_model.pkl",
    "gbm_model" : "models/gbm_model.pkl",
}

models = dict()

for model_name, model_path in model_paths.items():
    models[model_name] = pickle.load(open(model_path, 'rb'))
    

@app.route('/', methods=['GET', 'POST'])
def get_crop():
    crop = None
    if request.method == 'POST':
        try:
            # Fetch form data and ensure proper conversion to int
            data = request.get_json()
            N = int(data.get('N', 0))  # default to 0 if not provided
            P = int(data.get('P', 0))
            K = int(data.get('K', 0))
            selected_model_str = data.get('selectedModel', 'DecisionTreeClassifier')
            selected_model = models[selected_model_str]
            latitude = float(data.get('latitude', 0.0))
            longitude = float(data.get('longitude', 0.0))

            print(f"Received: N={N}, P={P}, K={K}, lat={latitude}, lon={longitude}")

            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid=08e7cb69def618692cd124606c70b9ae"
            response = requests.get(url)
            data = eval(response.content)
            weather = data['main']
            pprint.pprint(weather)
            temp = weather['temp']
            humidity = weather['humidity']
            
            test_input = np.array([N, P, K, temp-273.15, humidity, 6.033013, 200.098026]).reshape(1, 7)
            
            # Make prediction
            result = model.predict(test_input)
            
            # Pass prediction to the template
            crop = result[0]
            print(f"Predicted Crop: {crop}")
            return jsonify({"crop": crop}), 200
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"crop": "Error occurred during prediction"}), 500  # Return 500 on error
    
    return render_template('index.html', crop=crop, models = list(models.keys()))

if __name__ == '__main__':
    app.run(debug=True)
