from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.tree import export_text
import pydotplus
from matplotlib import pyplot as plt


#  the obesity dataset
data = pd.read_csv('/Users/sabiqadar/Desktop/zainab-portfolio/decision_treee_optimisation/ObesityDataSet_raw_and_data_sinthetic.csv')

# rename columns for better readability
data.rename(columns={
    'FAVC': 'HighCaloricFoodConsumption',
    'FCVC': 'VegetableConsumptionFrequency',
    'NCP': 'NumberOfMainMeals',
    'CAEC': 'EatingCaloriesDuringActivities',
    'CH2O': 'WaterConsumptionLiters',
    'SCC': 'CaloriesConsumptionMonitoring',
    'FAF': 'PhysicalActivityFrequency',
    'TUE': 'TimeUsingElectronicDevices',
    'CALC': 'AlcoholConsumption',
    'MTRANS': 'TransportationMethod',
    'NObeyesdad': 'WeightCategory'
}, inplace=True)


x = data.drop('WeightCategory', axis=1)
y = data['WeightCategory']

# One-hot encoder categorical features
categorical_features = x.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
x_encoded = pd.DataFrame(encoder.fit_transform(x[categorical_features]), columns=encoder.get_feature_names_out(categorical_features))
x = pd.concat([x.drop(categorical_features, axis=1).reset_index(drop=True), x_encoded.reset_index(drop=True)], axis=1)


# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train a decision tree classifier
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth =5, random_state=42)
clf.fit(x_train, y_train)

#  predictions
y_pred = clf.predict(x_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy before optimization: {accuracy:.4f}')

# display the tree using matplotlib
plt.figure(figsize=(20,15))
tree.plot_tree(
    clf,
    feature_names=x.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Obesity Decision Tree Flowchart", fontsize=20)
plt.show()