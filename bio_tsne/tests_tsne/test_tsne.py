import matplotlib.pyplot as plt
import numpy as np

xyz=np.array(np.random.random((5,2)))

color = np.array([55.3,55.4,55.9,60.1,60.5])

#print xyz
print color.shape
print xyz[:,2].shape 

marker_size=15
#plt.scatter(xyz[:,0],xyz[:,1],marker_size,c=xyz[:,2])
plt.scatter(xyz[:,0],xyz[:,1],marker_size,c=color)

print xyz[:,2].shape 

cbar=plt.colorbar()
cbar.set_label("elevation (m)",labelpad=-1)
 
plt.title("Point observations")
plt.xlabel("Easting")
plt.ylabel("Northing")
 
plt.show()
