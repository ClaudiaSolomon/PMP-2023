import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
x=stats.gamma(4,0,1/3)
y=stats.gamma(4,0,1/2)
z=stats.gamma(5,0,1/2)
w=stats.gamma(5,0,1/3)

latenta=stats.expon(0,1/4)

p1=0.25
p2=0.25
p3=0.3
p4=0.2


x_samples = x.rvs(size=10000)
y_samples = y.rvs(size=10000)
z_samples = z.rvs(size=10000)
w_samples = w.rvs(size=10000)
latency_samples = latenta.rvs(size=10000)

x_samples_with_latency = x_samples + latency_samples
y_samples_with_latency = y_samples + latency_samples
z_samples_with_latency = z_samples + latency_samples
w_samples_with_latency = w_samples + latency_samples

timp = p1*x_samples_with_latency+p2*y_samples_with_latency+p3*z_samples_with_latency+p4*w_samples_with_latency
X3 = np.mean(timp> 3)
az.plot_posterior({'X':X3})
plt.show()


