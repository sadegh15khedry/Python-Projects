import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.normal(5.0, 1.0, 10)
y = np.random.normal(5.0, 2.0, 10)
print(x)
print(y)
i1 = input("hist or scatter or lreg or preg? ")

if i1 == "hist":
    print("hist fo x is")
    plt.hist(x, 10)
    plt.show()
elif i1 == "scatter":
    plt.scatter(list(x), list(y))
    plt.show()
elif i1 == "lreg":
    regressor = LinearRegression()
    regressor.fit(x, y)
    # y_pred = regressor.predict(x)
    # plt.scatter(x, y)
    # plt.plot(x, regressor.predict(x))
    # plt.show()
elif i1 == "preg":
    myModel = np.poly1d(np.polyfit(x, y, 3))
    myLine = np.linspace(2, 95, 100)
    plt.scatter(x, y)
    plt.plot(myLine, myModel(myLine))
    plt.show()
else:
    print("input not valid")
