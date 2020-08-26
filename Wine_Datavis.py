import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#create the graph and visualization for the data
fig, ax = plt.subplots(1,2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red Wine')
ax[1].hist(white.alcohol, 10, facecolor='white', ec='black', lw=0.5, alpha=0.5, label='White Wine')

fig.subplots_adjust(left=0.2, right=0.5, bottom=0.2, top=0.5, hspace=0.05, wspace=0.5)
ax[0].set_ylim([0,1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")

fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()

#histogram as an alternative visual
print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

#visualize the Sulfates of the wine in a scatter plot format
fig, ax = plt.subplots(1,2,figsize=(8,4))

ax[0].scatter(red['quality'], red['sulphates'],color='red')
ax[1].scatter(white['quality'], white['sulphates'], color='white', edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()

#This scatter plot shows that the red wines tend to have more sulphates than the white





