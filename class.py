import numpy as np
          
a = [36.87, 6.08, 22.82, 0.66, 0.88, 1.23, 0.21, 0.55, 15.93, 1.16, 4.02, 1.22, 0.13, 7, 0.27, 0.23, 0.23, 0.09, 0.41]
a = [i/100.0 for i in a]

a = np.array(a)
np.save("Pseudo/class_distribution_gt.npy", a/100.0)