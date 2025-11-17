import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

#print(model)
yp= model.predict([[0, 0, 2.0, 2, 2018, 15000]])
print("Predicted value:", yp)