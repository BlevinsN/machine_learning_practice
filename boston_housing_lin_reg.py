from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
features = boston.data[:,0:2]
target = boston.target
regression = LinearRegression()
model = regression.fit(features, target)

print(model.predict(features)[0])
print(target[0])