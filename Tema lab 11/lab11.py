import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#1
clusters = 3 
n_cluster = [200, 150, 150] 
n_total = sum(n_cluster)
means = [5, 0, 1] 
std_devs = [2, 2, 2] 

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

az.plot_kde(np.array(mix))
plt.show()

#2
mix = mix.reshape(-1, 1)
num_components = [2, 3, 4]

for n_components in num_components:
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(mix)

    plt.figure(figsize=(8, 4))
    plt.hist(mix, bins=30, density=True, alpha=0.5, color='b')
    plt.title(f'{n_components} distributii gaussiene')
    
    x = np.linspace(mix.min(), mix.max(), 1000)
    pdf = np.exp(model.score_samples(x.reshape(-1, 1)))
    plt.plot(x, pdf, '-r')

    plt.show()

#3
log_likelihoods = []
samples = []
for n_components in num_components:
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(mix)
    log_likelihood = model.score_samples(mix)
    log_likelihoods.append(log_likelihood)
    samples.append(model.sample(1500)[0])  

idata = az.from_dict(
    log_likelihood={"mix": log_likelihoods},
    posterior_predictive={"mix": np.array(samples)},
    observed_data={"mix": mix.flatten()}
)

waic_results = az.waic(idata, pointwise=True, scale="deviance")
loo_results = az.loo(idata)

print("WAIC:", waic_results, "\nLOO:",loo_results)
