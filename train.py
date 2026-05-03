# train.py
# Run this once before starting the app.
# It trains the fraud detection model and saves the results.
# Command: python train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

# The 8 Beneish ratio column names
ratio_columns = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']

# Load the training data (1,036 labelled companies)
training_data = pd.read_csv('financial_data.csv')
print("Loaded", len(training_data), "companies")

# X = the 8 ratios (what the model learns from)
# y = 1 if Fraud, 0 if Legit (what the model learns to predict)
X = training_data[ratio_columns]
y = (training_data['Label'] == 'Fraud').astype(int)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("Model trained.")

# Evaluate using 5-fold cross validation
accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
auc_scores      = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

average_accuracy = round(accuracy_scores.mean(), 4)
average_auc      = round(auc_scores.mean(), 4)

print("Accuracy:", round(average_accuracy * 100, 2), "%")
print("AUC-ROC: ", round(average_auc  * 100, 2), "%")

# Confusion matrix on the test set
predictions    = model.predict(X_test)
cm             = confusion_matrix(y_test, predictions)
true_negative  = int(cm[0, 0])
false_positive = int(cm[0, 1])
false_negative = int(cm[1, 0])
true_positive  = int(cm[1, 1])

# Save model stats — app.py reads this to show on the dashboard
model_stats = pd.DataFrame([{
    'accuracy': average_accuracy,
    'auc_roc':  average_auc,
    'total':    len(training_data),
    'fraud':    int(y.sum()),
    'legit':    int((y == 0).sum()),
    'TN': true_negative,
    'FP': false_positive,
    'FN': false_negative,
    'TP': true_positive,
}])
model_stats.to_csv('model_info.csv', index=False)

# Save which ratios mattered most to the model
importance_data = pd.DataFrame({
    'Feature':    ratio_columns,
    'Importance': model.feature_importances_
})
importance_data = importance_data.sort_values('Importance', ascending=False)
importance_data.to_csv('feature_importance.csv', index=False)

print("Saved model_info.csv and feature_importance.csv")
print("Done. Now run: streamlit run app.py")
