import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pymc as pm
#1
mu, sigma = 0, 0.1 # valori alese de mine
timp_asteptare = np.random.normal(mu, sigma, 100) #generare 100 de timpi 
print(timp_asteptare)
print(timp_asteptare.mean())

#3

# with pm.Model() as model:
#     sigma = pm.Beta('sigma', alpha=1., beta=1.)
#     y = pm.Bernoulli('y', p=sigma)
#     timp_asteptare = pm.Normal("timp_asteptare", mu=y, sigma=sigma)

#     with model:
#         trace = pm.sample(100, cores=1)
#         az.plot_posterior(trace)
#         plt.show()

#2
