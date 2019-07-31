import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 

'''random = np.linspace(0.2, np.minimum(rv.dist.rvs(random_state=1), 0.45))

print(random)
print(rv.pdf(random))

plt.plot(random, rv.pdf(random))
plt.show()
'''

'''rv = stats.expon(scale=100)
random = np.linspace(0,  500, num=500)

print(random)
print(rv.pdf(random))

plt.plot(random, rv.pdf(random))
plt.title("'C' Probability Density Function")
plt.xlabel("Possible 'C' values")
plt.ylabel("Probability Density")
plt.show()'''
'''
rv = stats.expon(scale=.1)
random = np.linspace(0,  0.9, num=500)

print(random)
print(rv.pdf(random))

plt.plot(random, rv.pdf(random))
plt.title("Gamma Probability Density Function")
plt.xlabel("Possible Gamma values")
plt.ylabel("Probability Density")
plt.show()'''