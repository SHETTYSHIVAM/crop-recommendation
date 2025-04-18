from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import requests
import pprint
from dotenv import load_dotenv
import pandas as pd
import os
load_dotenv()

API_KEY = os.getenv('API_KEY')
app = Flask(__name__)

# # Load your model
# model = pickle.load(open('./crop.pkl', 'rb'))
crop_predictor = pickle.load(open('models/RandomForestClassifier.pkl', 'rb'))

# for model_name, model_path in model_paths.items():
#     models[model_name] = pickle.load(open(model_path, 'rb'))

fertilizer_predictor = pickle.load(open('models/fertilizer_recommendation.pkl', 'rb'))

CROP_NAMES = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

FERTILIZER_NAMES = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20',
       '10-26-26']

def get_weather_info(latitude, longitude):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}"
    response = requests.get(url)
    data = eval(response.content)
    weather = data['main']
    pprint.pprint(weather)
    temp = weather['temp']
    humidity = weather['humidity']
    return temp, humidity



@app.route('/crop', methods=['GET', 'POST'])
def get_crop():
    crop = None
    if request.method == 'POST':
        try:
            data = request.get_json()
            N = int(data.get('N', 0))
            P = int(data.get('P', 0))
            K = int(data.get('K', 0))
            pH = float(data.get('pH',7))
            latitude = float(data.get('latitude', 0.0))
            longitude = float(data.get('longitude', 0.0))

            print(f"Received: N={N}, P={P}, K={K}, lat={latitude}, lon={longitude}")

            temp, humidity = get_weather_info(latitude=latitude, longitude=longitude)
            
            test_input = np.array([N, P, K, temp-273.15, humidity, pH, 200.098026]).reshape(1, 7)
            
            result = crop_predictor.predict(test_input)
            probabilities = crop_predictor.predict_proba(test_input)
            top_n = data.get('top_n', 3)
            top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:]
            recommended_crops = [CROP_NAMES[i] for i in top_n_indices[0]]
            print(probabilities.tolist())
            for i, prob in enumerate(probabilities):
                print(f"Sample {i+1}: Class 0 Probability = {prob[0]:.4f}, Class 1 Probability = {prob[1]:.4f}")

            X_test = np.load('X_test.npy', allow_pickle=True)
            y_test = np.load('y_test.npy', allow_pickle=True)

            accuracy = crop_predictor.score(X_test, y_test)
            
            crop = result[0]
            print(f"Predicted Crop: {crop}")
            return jsonify({"crop": crop, "crops": recommended_crops, "accuracy": f"{accuracy*100:.2f}", "success": True}), 200
        except TypeError as e:
            print(f"Error during prediction: {e}")
            return jsonify({"success":False, "message": "Invalid Input"}), 400  # Return 500 on error
        except Exception as e :
            print(f"Error during prediction: {e}")
            return jsonify({"success":False, "message": "Server Error, Try again later"}), 500  # Return 500 on error
    return render_template('crop.html', crop=crop)


@app.route('/fertilizer', methods=['GET', 'POST'])
def get_fertilizer():
    if request.method == 'POST':
        # Moisture	Nitrogen	Potassium	Phosphorous Soil Type	Crop Type
        data = request.get_json()
        moisture = int(data.get('moisture', 0))
        N = int(data.get('nitrogen', 0))
        P = int(data.get('phosphorous', 0))
        K = int(data.get('potassium', 0))
        soil_type = (data.get('soilType', 'Clayey'))
        crop_type = (data.get('cropType', 'Paddy'))
        latitude = float(data.get('latitude', 0.0))
        longitude = float(data.get('longitude', 0.0))
        
        temp, humidity = get_weather_info(latitude, longitude)

        pprint.pprint(data)

        test_input = pd.DataFrame([[temp-273.15, humidity, moisture, N, K, P, soil_type, crop_type]],
                                  columns=['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium','Phosphorous', 'Soil Type', 'Crop Type'])

        result = fertilizer_predictor.predict(test_input)
        fertilizer = result[0]

        probabilities = fertilizer_predictor.predict_proba(test_input)
        top_n = data.get('top_n', 3)
        top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:]
        recommended_fertilizers = [FERTILIZER_NAMES[i] for i in top_n_indices[0]]

        print('Recommended Fertilizer: ', fertilizer)
        return jsonify({'success':True, 'fertilizer': fertilizer, 'fertilizers': recommended_fertilizers})
    else:
        return render_template('fertilizer.html')


if __name__ == '__main__':
    app.run(debug=True)