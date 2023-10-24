from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import expon

alfa=2.0

poisson_dist = poisson(mu=20)
normal_dist = norm(loc=2, scale=0.5)
exponential_dist = expon(scale=1.0 / alfa)

