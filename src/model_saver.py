import joblib

def save_model(model, file_path):
    joblib.dump(model, file_path)
