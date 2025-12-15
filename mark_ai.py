import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers, models


df = pd.read_csv("noten_beispiel.csv")
df['Datum'] = pd.to_datetime(df['Datum'])
df['Monat'] = (df['Datum']).dt.month
df['Year'] = (df['Datum']).dt.year

X = df[['Monat']]
y = df['Note'].values.astype(float)

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
loss, mae = model.evaluate(X_test, y_test)

print(f"accuracy: {mae}")
print(f"loss: {loss}")

y_pred = model.predict(X_test)

y_pred_all= model.predict(X)

plt.plot(X, y, color='blue', label='Tatsächliche von Noten', marker='o', markerfacecolor='yellow')
plt.plot(X, y_pred_all, color='red', linestyle='--', label='Vorhersage von Noten')
plt.title('Noten der Schüler')
plt.xlabel('Monat')
plt.ylabel('Noten')
plt.show()