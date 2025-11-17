import pandas as pd
import pickle
from datetime import datetime

# Load model + encoders
model = pickle.load(open("maintenance_model.pickle", "rb"))
encoders = pickle.load(open("encoders.pickle", "rb"))


last_service = "17-09-2023"
warranty_exp = "12-12-2027"

# Convert dates
today = pd.Timestamp.today()
days_since_service = (today - pd.to_datetime(last_service, dayfirst=True)).days
days_to_warranty = (pd.to_datetime(warranty_exp, dayfirst=True) - today).days

new_car = pd.DataFrame([{
    "Vehicle_Model": "SUV",
    "Mileage": 55000,
    "Maintenance_History": "Poor",
    "Reported_Issues": 2,
    "Vehicle_Age": 5,
    "Fuel_Type": "Petrol",
    "Transmission_Type": "Automatic",
    "Engine_Size": 1500,
    "Odometer_Reading": 55000,
    "Owner_Type": "First",
    "Insurance_Premium": 12000,
    "Service_History": 1,
    "Accident_History": 2,
    "Fuel_Efficiency": 10,
    "Tire_Condition": "Good",
    "Brake_Condition": "Good",
    "Battery_Status": "Good",
    "Days_Since_Service": days_since_service,
    "Days_To_Warranty_Expiry": days_to_warranty
}])


for col in encoders:
    new_car[col] = encoders[col].transform(new_car[col])


expected_cols = [
    'Vehicle_Model', 'Mileage', 'Maintenance_History', 'Reported_Issues',
    'Vehicle_Age', 'Fuel_Type', 'Transmission_Type', 'Engine_Size',
    'Odometer_Reading', 'Owner_Type', 'Insurance_Premium', 'Service_History',
    'Accident_History', 'Fuel_Efficiency', 'Tire_Condition', 'Brake_Condition',
    'Battery_Status', 'Days_Since_Service', 'Days_To_Warranty_Expiry'
]

new_car = new_car[expected_cols]


prediction = model.predict(new_car)

if prediction[0]:
    print("Needs Maintenance")
else:
    print("Does not need maintenance")
