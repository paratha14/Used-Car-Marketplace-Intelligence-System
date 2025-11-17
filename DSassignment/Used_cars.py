import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r"C:\Users\3win\Desktop\VS code files\vs code py\DSassignment\vehicle_maintenance_data.csv")
df.dropna()
df.drop_duplicates()
df=df[df["Vehicle_Model"].isin(["Car","SUV"])]


df["Last_Service_Date"] = pd.to_datetime(df["Last_Service_Date"], dayfirst=True)
df["Warranty_Expiry_Date"] = pd.to_datetime(df["Warranty_Expiry_Date"], dayfirst=True)

today = pd.Timestamp.today()

df["Days_Since_Service"] = (today - df["Last_Service_Date"]).dt.days
df["Days_To_Warranty_Expiry"] = (df["Warranty_Expiry_Date"] - today).dt.days

df.drop(["Last_Service_Date", "Warranty_Expiry_Date"], axis=1, inplace=True)

cols=['Vehicle_Model', 'Maintenance_History',
       'Fuel_Type', 'Transmission_Type',
       'Owner_Type', 'Tire_Condition',
       'Brake_Condition', 'Battery_Status']
encoders = {}
for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  
    encoders[col] = le
    

x=df.drop("Need_Maintenance", axis=1)
y=df['Need_Maintenance']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,         
    max_depth=None,          
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


with open("maintenance_model.pickle", "wb") as f:
    pickle.dump(rf, f)

with open("encoders.pickle", "wb") as f:
    pickle.dump(encoders, f)