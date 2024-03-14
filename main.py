import pandas as pd
from preprocess import preprocess_data
# Assuming the preprocessing function is defined above or imported

def main():
    # Load your dataset
    df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
    df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df['Blood Pressure'].str.split('/', expand= True)
    df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']].apply(pd.to_numeric)


    # Apply preprocessing
    X_sm, y_sm = preprocess_data(df, 'Sleep Disorder' ,['Person ID', 'Blood Pressure'])
    print(X_sm)
    print(y_sm)
    
    # Here you can further process, save, or directly use your preprocessed data
    # Example: Save the preprocessed features to a CSV (adjust as necessary)
    # pd.DataFrame(features_preprocessed).to_csv('preprocessed_features.csv', index=False)

if __name__ == "__main__":
    main()