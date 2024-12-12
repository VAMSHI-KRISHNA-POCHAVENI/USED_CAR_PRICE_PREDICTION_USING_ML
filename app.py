from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('xg_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        fuel_type = int(request.form['Fuel_Type'])
        seller_type = int(request.form['Seller_Type'])
        transmission = int(request.form['Transmission'])
        owner = int(request.form['Owner'])
        age = int(request.form['Age'])

        # Prepare data for prediction
        features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, age]])
        
        # Predict the selling price
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        
        return render_template('result.html', prediction_text=f'Estimated Selling Price: ₹ {output}')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
