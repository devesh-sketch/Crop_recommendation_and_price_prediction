# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import uvicorn
# import os # Added for port flexibility

# app = FastAPI(title="Crop Recommendation API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Using relative paths is correct for Render
# try:
#     model = joblib.load('crop_model.pkl')
#     label_encoder = joblib.load('label_encoder.pkl')
#     print("Models loaded successfully.")
# except Exception as e:
#     print(f"Error loading models: {e}")

# CROP_PRICES = {
#     "rice": 2183, "maize": 2090, "chickpea": 5335, "kidneybeans": 6500,
#     "pigeonpeas": 7000, "mothbeans": 5800, "mungbean": 7755, "blackgram": 6600,
#     "lentil": 6000, "watermelon": 1500, "muskmelon": 2000, "cotton": 6620, "jute": 5050
# }

# class ClimateData(BaseModel):
#     temp: float
#     humidity: float
#     rainfall: float

# @app.post("/api/recommend")
# async def predict(data: ClimateData):
#     try:
#         input_df = pd.DataFrame([{
#             "temperature": data.temp,
#             "humidity": data.humidity,
#             "rainfall": data.rainfall
#         }])
        
#         prediction_numeric = model.predict(input_df)
#         raw_name = label_encoder.inverse_transform(prediction_numeric)[0]
#         crop_name = str(raw_name).lower().strip()
#         price_val = CROP_PRICES.get(crop_name, 0)
        
#         return {
#             "status": "success",
#             "crop": crop_name.capitalize(), 
#             "price_value": price_val
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# if __name__ == "__main__":
#     # Get port from environment or default to 5000
#     port = int(os.environ.get("PORT", 5000)) 
#     uvicorn.run(app, host="0.0.0.0", port=port)



import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# Enable CORS for all routes (equivalent to your FastAPI middleware)
CORS(app)

# Load the machine learning models
try:
    model = joblib.load('crop_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

CROP_PRICES = {
    "rice": 2183, "maize": 2090, "chickpea": 5335, "kidneybeans": 6500,
    "pigeonpeas": 7000, "mothbeans": 5800, "mungbean": 7755, "blackgram": 6600,
    "lentil": 6000, "watermelon": 1500, "muskmelon": 2000, "cotton": 6620, "jute": 5050
}

@app.route("/api/recommend", methods=["POST"])
def predict():
    try:
        # Get JSON data from the incoming request
        data = request.get_json()
        
        # Extract fields (matching your original schema: temp, humidity, rainfall)
        temp = float(data.get("temp"))
        humidity = float(data.get("humidity"))
        rainfall = float(data.get("rainfall"))
        
        # Create DataFrame for model prediction
        input_df = pd.DataFrame([{
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall
        }])
        
        # Predict using the loaded model
        prediction_numeric = model.predict(input_df)
        raw_name = label_encoder.inverse_transform(prediction_numeric)[0]
        crop_name = str(raw_name).lower().strip()
        price_val = CROP_PRICES.get(crop_name, 0)
        
        # Return JSON response
        return jsonify({
            "status": "success",
            "crop": crop_name.capitalize(), 
            "price_value": price_val
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Dynamically bind to the PORT provided by Render, defaulting to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)