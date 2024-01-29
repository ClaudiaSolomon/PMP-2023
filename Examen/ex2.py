import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#a
def posterior_grid(grid_points=50, heads_obtinut_la_aruncarea=20):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/5*heads_obtinut_la_aruncarea, grid_points)
    #probabilitatea de a obtine o stema e 1/5^a
    likelihood = stats.geom.pmf(5, heads_obtinut_la_aruncarea)
    #n=5
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

points = 10
grid, posterior = posterior_grid(points, 20)

plt.plot(grid, posterior, 'o-')
plt.yticks([])
plt.xlabel('θ')
plt.show()

#b
MAP_estimate = grid[np.argmax(posterior)]
print("Prob a posteriori maxima pentru theta: ", MAP_estimate)


