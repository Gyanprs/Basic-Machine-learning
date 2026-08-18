import numpy as np
from sklearn.linear_model import LinearRegression

# Pizza sizes and prices 
sizes = np.array([2, 10, 12, 14]).reshape(-1, 1)
prices = np.array([3, 9, 12, 15])

# Create and train the linear regression model
model = LinearRegression()
model.fit(sizes, prices)  

#Preditcion 
prediction = model.predict([[8]])
print(f"Predicted price: ${prediction[0]:.2f}")
