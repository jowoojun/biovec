import numpy as np
import matplotlib.pyplot as plt



colors = np.array(100)
print colors
for i in range(10):
    plt.scatter(x=np.array(i), y=np.array(1))
plt.show()
