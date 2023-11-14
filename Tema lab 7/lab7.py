import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
file_path = 'C:\\Users\\Claudia\\Downloads\\auto-mpg.csv'  
data = pd.read_csv(file_path)
mpg=data['mpg'].values
# CP=data['horsepower'].values
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
CP = data['horsepower'].values

data_clean = data.dropna(subset=['horsepower', 'mpg'])

plt.scatter(data_clean['horsepower'], data_clean['mpg'], color='blue')
plt.xlabel('CP')
plt.ylabel('MPG')
# plt.show()
if __name__ == '__main__':
    with pm.Model() as model_g:
        a = pm.Normal('a', mu=0, sigma=10)
        β = pm.Normal('β', mu=0, sigma=1)
        ε = pm.HalfCauchy('ε', 5)
        μ=a + β * data_clean['horsepower']

        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=data_clean['mpg'])
        idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
        az.plot_trace(idata_g, var_names=['a', 'β', 'ε'])

    # az.plot_pair(idata_g, var_names=['a', 'β'], scatter_kwargs={'alpha': 0.1})

        plt.plot(CP, mpg, 'C0.')
        posterior_g = idata_g.posterior.stack(samples={"chain", "draw"})
        alpha_m = posterior_g['a'].mean().item()
        beta_m = posterior_g['β'].mean().item()
        draws = range(0, posterior_g.samples.size, 10)
        plt.plot(CP, posterior_g['a'][draws].values+ posterior_g['β'][draws].values * CP[:,None],
        c='gray', alpha=0.5)
        plt.plot(CP, alpha_m + beta_m * CP, c='k',
        label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
        plt.xlabel('x')
        plt.ylabel('y', rotation=0)
        plt.legend()

        plt.plot(CP, alpha_m + beta_m * CP, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
        sig = az.plot_hdi(CP, posterior_g['μ'].T, hdi_prob=0.95, color='k')
        plt.xlabel('x')
        plt.ylabel('y', rotation=0)
        plt.legend()
    
        plt.show()

# cu cat HDI e mai ingust cu atat suntem mai siguri pe predictiile facute