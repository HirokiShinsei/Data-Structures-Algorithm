import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1 - Import Dataset
data = pd.read_csv('Mall_Customers.csv')

# Step 2 - Select AnnualIncome and Spending Score columns
data = data[['AnnualIncome (k$)', 'Spending Score (1-100)']]

# Standardization
data['AnnualIncome (k$)'] = (data['AnnualIncome (k$)'] - data['AnnualIncome (k$)'].mean()) / data['AnnualIncome (k$)'].std()
data['Spending Score (1-100)'] = (data['Spending Score (1-100)'] - data['Spending Score (1-100)'].mean()) / data['Spending Score (1-100)'].std()

# Convert data to a NumPy array
data = data.values
x = data[:, 0]
y = data[:, 1]

k = 5
cx = [-1, 0, -1, 3, -1.5]  # Initial mean of x
cy = [1, 2, -1, 2, 1]  # Initial mean of y
mean_oldx = cx.copy()
mean_newx = cx.copy()
mean_oldy = cy.copy()
mean_newy = cy.copy()

outputx = [[] for _ in range(k)]
outputy = [[] for _ in range(k)]
temp = 0

# Create a figure and axis
fig, ax = plt.subplots()
ax.scatter(x, y, c='blue', marker='o')
scatter = ax.scatter(cx, cy, c='red', marker='x', s=200)
lines = []

def animate(frame):
    global cx, cy, outputx, outputy, mean_oldx, mean_oldy, mean_newx, mean_newy, temp
    if temp == 0:
        ax.clear()
        ax.scatter(x, y, c='blue', marker='o')
        mean_oldx = mean_newx.copy()
        mean_oldy = mean_newy.copy()

        for ij in range(len(x)):
            mina = []
            mu = x[ij]
            nu = y[ij]

            for mk in range(len(cx)):
                mina.append(np.sqrt((mu - cx[mk]) ** 2 + (nu - cy[mk]) ** 2))

            mina = np.array(mina)
            index = np.argmin(mina)
            gc = mina[index]

            outputx[index].append(mu)
            outputy[index].append(nu)

            # Draw lines connecting data points to centroids
            line = ax.plot([mu, cx[index]], [nu, cy[index]], color='gray')
            lines.extend(line)

        gmckx = []
        gmcky = []

        for i in range(k):
            gmckx.append(np.mean(outputx[i]))
            gmcky.append(np.mean(outputy[i]))

        cx = gmckx
        cy = gmcky
        mean_newx = cx.copy()
        mean_newy = cy.copy()
        gum = 0
        bum = 0

        if np.array_equal(mean_newx, mean_oldx) and np.array_equal(mean_newy, mean_oldy):
            temp = 1
        else:
            outputx = [[] for _ in range(k)]

        # Update scatter plot
        scatter = ax.scatter(cx, cy, c='red', marker='x', s=200)

# Create an animation
ani = FuncAnimation(fig, animate, frames=100, repeat=False)

plt.show()
