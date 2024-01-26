#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define colors for each fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
persons = ['Farrah', 'Fred', 'Felicia']

# Plotting the stacked bar graph
fig, ax = plt.subplots()
for i, person in enumerate(persons):
    bottom = np.zeros(len(fruit_names))
    for j, fruit_name in enumerate(fruit_names):
        ax.bar(person, fruit[j, i], width=0.5, color=colors[j], bottom=bottom, label=fruit_name)
        bottom += fruit[j, i]

# Adding labels and title
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

# Show the plot
plt.show()
