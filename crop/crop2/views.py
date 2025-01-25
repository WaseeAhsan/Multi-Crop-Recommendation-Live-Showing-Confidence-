import pandas as pd
import numpy as np
from django.shortcuts import render
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset and train the model globally to avoid retraining on every request
df = pd.read_csv('crop2/Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the primary XGBoost model
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X, y_encoded)

def home(request):
    return render(request, 'home.html')

def crop_recommendation(request):
    if request.method == 'POST':
        try:
            # Extract user input from the form
            N = float(request.POST.get('N'))
            P = float(request.POST.get('P'))
            K = float(request.POST.get('K'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            ph = float(request.POST.get('ph'))
            rainfall = float(request.POST.get('rainfall'))

            # Prepare input data for prediction
            input_data = pd.DataFrame([{
                "N": N, "P": P, "K": K,
                "temperature": temperature, "humidity": humidity,
                "ph": ph, "rainfall": rainfall
            }])

            # Predict probabilities for the primary model
            y_prob = xgb_model.predict_proba(input_data)[0]
            sorted_classes = np.argsort(y_prob)[::-1]

            # First best crop
            first_best_crop = label_encoder.inverse_transform([sorted_classes[0]])[0]
            first_best_confidence = y_prob[sorted_classes[0]] * 100

            # Remove the first-best crop from the training data
            mask = y_encoded != sorted_classes[0]
            second_best_X_train = X[mask]
            second_best_y_train = y_encoded[mask]

            # Train a secondary model for the second-best crop
            second_best_encoder = LabelEncoder()
            second_best_y_train_encoded = second_best_encoder.fit_transform(second_best_y_train)

            second_best_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
            second_best_model.fit(second_best_X_train, second_best_y_train_encoded)

            # Predict second-best probabilities
            second_best_prob = second_best_model.predict_proba(input_data)[0]
            second_best_sorted = np.argsort(second_best_prob)[::-1]

            # Decode the second-best crop
            second_best_crop_encoded = second_best_encoder.inverse_transform([second_best_sorted[0]])[0]
            second_best_crop = label_encoder.inverse_transform([second_best_crop_encoded])[0]
            second_best_confidence = second_best_prob[second_best_sorted[0]] * 100

            # Return results to the template
            return render(request, 'home.html', {
                "input_data": input_data.to_dict(orient='records')[0],
                "first_best_crop": first_best_crop,
                "first_best_confidence": f"{first_best_confidence:.2f}%",
                "second_best_crop": second_best_crop,
                "second_best_confidence": f"{second_best_confidence:.2f}%"
            })

        except Exception as e:
            return render(request, 'home.html', {"error": str(e)})

    return render(request, 'home.html')
