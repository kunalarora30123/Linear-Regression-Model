#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model (Ensure you have trained and saved a model as 'house_price_model.pkl')
model = joblib.load("house_price_model.pkl")

@app.route('/')
def index():
    """Render the index.html page from the templates folder."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests via JSON input."""
    try:
        data = request.json
        features = [
            data['CRIM'], data['ZN'], data['INDUS'], data['CHAS'], data['NOX'],
            data['RM'], data['AGE'], data['DIS'], data['RAD'], data['TAX'],
            data['PTRATIO'], data['B'], data['LSTAT']
        ]
        prediction = model.predict([np.array(features)])
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




