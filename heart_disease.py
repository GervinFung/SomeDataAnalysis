def read_csv(path):
    import pandas as pd

    try:
        return pd.read_csv(path, encoding='ansi')
    except LookupError:
        return pd.read_csv(path, encoding='cp1252')


def clean_heart_disease_data():
    heart_disease_data = read_csv('heart.csv')

    heart_disease_data.drop_duplicates(inplace=True)
    heart_disease_data.reset_index(inplace=True, drop=True)
    heart_disease_data = heart_disease_data.dropna(how='all', axis=1)

    heart_disease_data.columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol',
                                  'Fasting Blood Sugar', 'Resting ECG', 'Max. HR Achieved', 'Exercise Induced Angina',
                                  'ST Depression', 'ST Slope', 'Num. Major Blood Vessels', 'Thalassemia', 'Condition']

    heart_disease_data['Condition'] = heart_disease_data['Condition'].apply(lambda x: 1 if x == 0 else 0)

    return heart_disease_data


def print_heart_disease_data_info(cleaned_heart_disease_data):
    print(cleaned_heart_disease_data.head())
    print(cleaned_heart_disease_data.shape)
    print(cleaned_heart_disease_data.describe())


def get_x_and_y(cleaned_heart_disease_data):
    x = cleaned_heart_disease_data[['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol',
                                  'Fasting Blood Sugar', 'Resting ECG', 'Max. HR Achieved', 'Exercise Induced Angina',
                                  'ST Depression', 'ST Slope', 'Num. Major Blood Vessels', 'Thalassemia']]

    y = cleaned_heart_disease_data['Condition']

    return x, y


def train_set_test_set(x, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size=0.2, random_state=42)


def classifier_model(x, y):
    import classifier
    x_train, x_test, y_train, y_test = train_set_test_set(x, y)
    classifier.all_classifier(x_train, x_test, y_train, y_test)


def clustering_model(x, y):
    import clustering
    clustering.all_clustering(x)


if __name__ == '__main__':
    cleaned_heart_disease_data = clean_heart_disease_data()
    print_heart_disease_data_info(cleaned_heart_disease_data)
    x, y = get_x_and_y(cleaned_heart_disease_data)
    # classifier_model(x, y)
    clustering_model(x, y)
