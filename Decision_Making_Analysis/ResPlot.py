import matplotlib.pyplot as plt

mode_labels = ['Hunting', 'Grazing']
estimation_labels = ['Correct', 'Fail']

# For task with all data (case 1)
mode_size = [0.028, 1-0.028]
hunt_cr = [0.015,1-0.015]
explode = (0.2, 0.0)

plt.subplot(1, 2, 1)
plt.pie(mode_size, explode=explode, labels=mode_labels,autopct='%1.1f%%',shadow=False, startangle=90)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.pie(hunt_cr, explode=explode, labels=estimation_labels,autopct='%1.1f%%',shadow=False, startangle=90)
plt.axis('equal')

plt.show()