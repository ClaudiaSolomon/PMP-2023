import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pymc as pm
import copy
#1
mu, sigma = 0, 0.1 # valori alese de mine
meanlist = []
for i in range(100):
    timp_asteptare = np.random.normal(mu, sigma,10).mean()
    meanlist.append(copy.deepcopy(timp_asteptare))
print(meanlist)

#2
#distributia beta este conjugata a posteriori a distributiei bernoulli
with pm.Model() as model:
    sigma = pm.Beta('sigma', alpha=1., beta=1.)
    y = pm.Bernoulli('y', p=sigma)
    timp_asteptare = pm.Normal("timp_asteptare", mu=y, sigma=sigma)
    observation = pm.Poisson("obs", mu=timp_asteptare, observed=meanlist)


    with model:
        trace = pm.sample(1000, cores=1)
        az.plot_posterior(trace)
        plt.show()

#3
with pm.Model() as model:
        sigma = pm.Beta('sigma', alpha=1., beta=1.)
        y = pm.Bernoulli('y', p=sigma)
        timp_asteptare = pm.Normal("timp_asteptare", mu=y, sigma=sigma)
        #estimare y
        trace = pm.sample(2000, tune=1000, chains=2)
        az.plot_posterior(trace)