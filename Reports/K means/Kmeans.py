# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # # Step 1 - Import Dataset
# # data = pd.read_csv('Mall_Customers.csv')

# # print(data.head())

# # # Step 2 - Select AnnualIncome and SpendingScore columns (4 and 5)
# # data = data.iloc[:, [3, 4]]

# # # Standardization
# # data['AnnualIncome'] = (data['AnnualIncome'] - data['AnnualIncome'].mean()) / data['AnnualIncome'].std()
# # data['SpendingScore'] = (data['SpendingScore'] - data['SpendingScore'].mean()) / data['SpendingScore'].std()

# # # Convert data to a NumPy array
# # data = data.values
# # x = data[:, 0]
# # y = data[:, 1]
# # plt.scatter(x, y, c='blue', marker='o')

# # Import Dataset
# data = pd.read_csv('Mall_Customers.csv')

# # Select AnnualIncome and Spending Score columns
# data = data[['AnnualIncome (k$)', 'Spending Score (1-100)']]

# # Standardization
# data['AnnualIncome (k$)'] = (data['AnnualIncome (k$)'] - data['AnnualIncome (k$)'].mean()) / data['AnnualIncome (k$)'].std()
# data['Spending Score (1-100)'] = (data['Spending Score (1-100)'] - data['Spending Score (1-100)'].mean()) / data['Spending Score (1-100)'].std()

# # Convert data to a NumPy array
# data = data.values
# x = data[:, 0]
# y = data[:, 1]

# plt.scatter(x, y, c='blue', marker='o')

# k = 5
# cx = [-1, 0, -1, 3, -1.5]  # initial mean of x
# cy = [1, 2, -1, 2, 1]  # initial mean of y
# mean_oldx = cx.copy()
# mean_newx = cx.copy()
# mean_oldy = cy.copy()
# mean_newy = cy.copy()
# outputx = [[] for _ in range(k)]
# outputy = [[] for _ in range(k)]
# temp = 0

# plt.scatter(cx, cy, c='red', marker='x', s=200)

# while temp == 0:
#     plt.pause(3)
#     plt.clf()
#     plt.scatter(x, y, c='blue', marker='o')
#     plt.scatter(cx, cy, c='red', marker='x', s=200)
#     mean_oldx = mean_newx.copy()
#     mean_oldy = mean_newy.copy()

#     for ij in range(len(x)):
#         mina = []
#         mu = x[ij]
#         nu = y[ij]

#         for mk in range(len(cx)):
#             mina.append(np.sqrt((mu - cx[mk]) ** 2 + (nu - cy[mk]) ** 2))

#         gc, index = min(mina)
#         plt.plot([x[ij], cx[index]], [y[ij], cy[index]], color='gray')
#         plt.scatter(cx, cy, c='red', marker='x', s=200)
#         outputx[index].append(mu)
#         outputy[index].append(nu)

#     plt.pause(0.5)
#     gmckx = []
#     gmcky = []

#     for i in range(k):
#         gmckx.append(np.mean(outputx[i]))
#         gmcky.append(np.mean(outputy[i]))

#     cx = gmckx
#     cy = gmcky
#     mean_newx = cx.copy()
#     mean_newy = cy.copy()
#     gum = 0
#     bum = 0

#     if np.array_equal(mean_newx, mean_oldx):
#         gum = 1

#     if np.array_equal(mean_newy, mean_oldy):
#         bum = 1

#     if gum == 1 and bum == 1:
#         temp = 1
#     else:
#         outputx = [[] for _ in range(k)]
#         outputy = [[] for _ in range(k)]

# plt.show()

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
