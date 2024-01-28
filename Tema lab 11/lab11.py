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

# #2
#     mix = np.array(mix)

#     clusters = [2, 3, 4]
#     models = []
#     idatas = []
#     for cluster in clusters:
#         with pm.Model() as model:
#             p = pm.Dirichlet('p', a=np.ones(cluster))
#             means = pm.Normal('means',
#                             mu=np.linspace(mix.min(), mix.max(), cluster),
#                             sigma=10, shape=cluster,
#                             transform=pm.distributions.transforms.ordered)
#             sd = pm.HalfNormal('sd', sigma=100)
#             y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
#             idata = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
#         idatas.append(idata)
#         models.append(model)

#     #3
#     comp_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
#     comp_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="loo", scale="deviance")

#     az.plot_compare(comp_waic)
#     az.plot_compare(comp_loo)
