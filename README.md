# Used Car Price & Maintenance Predictor

A minimalist ML project featuring **two machine learning models**:  
1. Used Car Price Prediction  
2. Car Maintenance Requirement Classification  

---

## Features

### **Used Car Price Predictor**
Predicts the estimated selling price of a used car based on:

- Manufacturer  
- Model  
- Engine Size  
- Fuel Type  
- Year of Manufacture  
- Mileage  

**Model Used:** Random Forest Regressor  
**Performance:** **R² Score — 0.9981**

---

### **Car Maintenance Requirement Classifier**
Predicts whether a car needs maintenance using:

- Vehicle Model  
- Maintenance History  
- Fuel Type  
- Transmission Type  
- Owner Type  
- Tire Condition  
- Brake Condition  
- Battery Status  

**Model Used:** RandomForest Classifier (or similar)  
**Performance:** **Accuracy — 0.99**

---

## Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-Learn  
- Jupyter Notebook / Google Colab  

---

## Workflow

1. Load and preprocess datasets  
2. Train the regression and classification models separately  
3. Input car details through script/notebook  
4. Receive:  
   - 💰 Predicted car price  
   - 🔧 Maintenance requirement (Yes/No)

---
