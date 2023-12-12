import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('C:\\Users\\Claudia\\Desktop\\an3\\PMP\\dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

# 1
with pm.Model() as model_p:
    a = pm.Normal('a', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = a + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_p_sd_10 = pm.sample(2000, return_inferencedata=True)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p_sd_10)
    plt.title('Model cu sd=10')

# 2
with pm.Model() as model_p_sd_100:
    a = pm.Normal('a', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = a + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_p_sd_100 = pm.sample(2000, return_inferencedata=True)

    plt.subplot(3, 1, 2)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p_sd_100)
    plt.title('Model cu sd=100')

with pm.Model() as model_p_sd_array:
    a = pm.Normal('a', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = a + pm.math.dot(β, x_1s)
    y = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_p_sd_array = pm.sample(2000, return_inferencedata=True)

    plt.subplot(3, 1, 3)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p_sd_array)
    plt.title('Model cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])')


plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
