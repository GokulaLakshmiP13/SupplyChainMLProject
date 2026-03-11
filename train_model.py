import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')

# Select features
features = [
    'Days for shipment (scheduled)',
    'Order Item Quantity',
    'Order Item Product Price',
    'Benefit per order',
    'Sales',
    'Order Item Discount Rate',
    'Late_delivery_risk'
]

data = data[features]

X = data.drop('Late_delivery_risk', axis=1)
y = data['Late_delivery_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully.")