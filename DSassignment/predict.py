import pandas as pd
import pickle
from datetime import datetime

model = pickle.load(open("maintenance_model.pickle", "rb"))
encoders = pickle.load(open("encoders.pickle", "rb"))

print("\n--- Vehicle Maintenance Prediction ---\n")

vehicle_model = input("Enter Vehicle Model (e.g. SUV, Car): ")
mileage = float(input("Enter Mileage (km): "))
maintenance_history = input("Enter Maintenance History (Good/Average/Poor): ")
reported_issues = int(input("Enter Number of Reported Issues: "))
vehicle_age = int(input("Enter Vehicle Age (years): "))
fuel_type = input("Enter Fuel Type (Petrol/Diesel/Electric): ")
transmission_type = input("Enter Transmission Type (Automatic/Manual): ")
engine_size = float(input("Enter Engine Size (cc): "))
odometer_reading = float(input("Enter Odometer Reading (km): "))
owner_type = input("Enter Owner Type (First/Second/Third): ")
insurance_premium = float(input("Enter Insurance Premium Amount: "))
service_history = int(input("Enter Service History Count: "))
accident_history = int(input("Enter Number of Accidents: "))
fuel_efficiency = float(input("Enter Fuel Efficiency (kmpl): "))
tire_condition = input("Enter Tire Condition (New/Good/Bad): ")
brake_condition = input("Enter Brake Condition (New/Good/Bad): ")
battery_status = input("Enter Battery Status (Good/Weak/New): ")
last_service = input("Enter Last Service Date (DD-MM-YYYY): ")
warranty_exp = input("Enter Warranty Expiry Date (DD-MM-YYYY): ")

today = pd.Timestamp.today()
days_since_service = (today - pd.to_datetime(last_service, dayfirst=True)).days
days_to_warranty = (pd.to_datetime(warranty_exp, dayfirst=True) - today).days

new_car = pd.DataFrame([{
    "Vehicle_Model": vehicle_model,
    "Mileage": mileage,
    "Maintenance_History": maintenance_history,
    "Reported_Issues": reported_issues,
    "Vehicle_Age": vehicle_age,
    "Fuel_Type": fuel_type,
    "Transmission_Type": transmission_type,
    "Engine_Size": engine_size,
    "Odometer_Reading": odometer_reading,
    "Owner_Type": owner_type,
    "Insurance_Premium": insurance_premium,
    "Service_History": service_history,
    "Accident_History": accident_history,
    "Fuel_Efficiency": fuel_efficiency,
    "Tire_Condition": tire_condition,
    "Brake_Condition": brake_condition,
    "Battery_Status": battery_status,
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

print("\n--- Prediction Result ---")
if prediction[0]:
    print("The vehicle Needs Maintenance.")
else:
    print("The vehicle Does NOT need maintenance.")
