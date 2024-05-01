from src import data_loader, preprocess, model_trainer, model_saver, model_evaluator
from sklearn.model_selection import train_test_split

def main():
    # Load data
    data_path = "data/Customer-Churn.csv"  # Provide the path to your CSV file
    data = data_loader.load_data(data_path)

    # Split data into features and target
    x,y = preprocess.preprocess_data(data)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 40, stratify=y)
    print(x.dtypes)
    # Train model
    model = model_trainer.train_model(x_train, y_train)
    accuracy = model_evaluator.evaluate_model(model,x_test,y_test)
    print("Model Accuracy: ",accuracy)
    
    # Save model
    model_path = "models/model.pkl"  # Provide the path to save the model
    model_saver.save_model(model, model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()