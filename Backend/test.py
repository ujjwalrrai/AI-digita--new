import joblib
model = joblib.load("./model/campaign_model.pkl")
print(type(model))
