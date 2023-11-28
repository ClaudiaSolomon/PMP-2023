import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

data = pd.read_csv("C:\\Users\\Claudia\\Desktop\\an3\\PMP\\Prices.csv")
pret_vanzare=data['Price'].values.astype(float)
frecventa_procesor=data['Speed'].values.astype(float)
marime_harddrive=data['HardDrive'].values.astype(float)

#1
if __name__ == '__main__':
    with pm.Model() as model_regression:       
            alfa = pm.Normal('alfa', mu=0, sigma=10)
            beta1 = pm.Normal('beta1', mu=0, sigma=1)
            beta2 = pm.Normal('beta2', mu=0, sigma=1)
            sigma = pm.HalfCauchy('sigma', 5)

            niu = pm.Deterministic('niu', alfa + beta1 * frecventa_procesor + beta2 * np.log(marime_harddrive))
            y_pred = pm.Normal('y_pred', mu=niu, sigma=sigma, observed=pret_vanzare)

            idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta1', 'beta2', 'sigma'])
    plt.show()

#2
    # posterior_data = idata['posterior']
    # alfa_m = posterior_data['alfa'].mean().item()
    # beta1_m = posterior_data['beta1'].mean().item()
    # beta2_m = posterior_data['beta2'].mean().item()
    # print(alfa_m, beta1_m,beta2_m)

    # plt.scatter(frecventa_procesor, marime_harddrive, marker='o')
    # plt.xlabel('frecventa_procesor')
    # plt.ylabel('marime_harddrive')
    # plt.plot(frecventa_procesor,  alfa_m + beta1_m * frecventa_procesor + beta2_m * np.log(marime_harddrive), c='k')
    # ppc = pm.sample_posterior_predictive(idata, model=model_regression)
    # posterior_predictive = ppc['posterior_predictive']
    # az.plot_hdi(frecventa_procesor, posterior_predictive['y_pred'], hdi_prob=0.95, color='gray', smooth=False)
    # plt.show()

    summary = az.summary(idata, var_names=['beta1', 'beta2'], hdi_prob=0.95)
    print(summary)

#3
#interval incredere 2.5%-97.5%
    is_beta1_significant = (summary['hdi_2.5%']['beta1'] > 0) or (summary['hdi_97.5%']['beta1'] < 0)
    is_beta2_significant = (summary['hdi_2.5%']['beta2'] > 0) or (summary['hdi_97.5%']['beta2'] < 0)

    if is_beta1_significant:
        print("Frecventa procesorului este un predictor semnificativ al pretului de vanzare")
    else:
        print("Frecventa procesorului nu este un predictor semnificativ al pretului de vanzare")

    if is_beta2_significant:
        print("Marimea hard diskului este un predictor semnificativ al pretului de vanzare")
    else:
        print("Marimea hard diskului nu este un predictor semnificativ al pretului de vanzare")

#4

    alfa_samples = idata.posterior['alfa'].values
    beta1_samples = idata.posterior['beta1'].values
    beta2_samples = idata.posterior['beta2'].values

    selected_alfa_samples = np.random.choice(alfa_samples.flatten(), size=5000, replace=True)
    selected_beta1_samples = np.random.choice(beta1_samples.flatten(), size=5000, replace=True)
    selected_beta2_samples = np.random.choice(beta2_samples.flatten(), size=5000, replace=True)

    new_frecventa_procesor = 33
    new_marime_harddrive = 540
    simulated_prices = (
        selected_alfa_samples +
        selected_beta1_samples * new_frecventa_procesor +
        selected_beta2_samples * np.log(new_marime_harddrive)
    )

    hdi_90_simulated = az.hdi(simulated_prices, hdi_prob=0.9)

    print(f"Intervalul de 90% HDI pentru prețul de vânzare simulat de 5000 ori este: "
          f"[{hdi_90_simulated[0]:.2f}, {hdi_90_simulated[1]:.2f}]")
    
#5
    ppc = pm.sample_posterior_predictive(idata, model=model_regression)
    posterior_predictive = ppc['y_pred']

    hdi_90_posterior_predictive = az.hdi(posterior_predictive.flatten(), hdi_prob=0.9)

    print(f"Intervalul de 90% HDI pentru distribuția predictivă posterioară este: "
          f"[{hdi_90_posterior_predictive[0]:.2f}, {hdi_90_posterior_predictive[1]:.2f}]")
    
#bonus: depinde de cerintele clientului si de coeficientul asociat variabilelei (daca e semnificativ diferit de 0)