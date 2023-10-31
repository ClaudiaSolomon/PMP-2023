
import pymc3 as pm
import pandas as pd
import numpy as np

file_path = 'C:\\Users\\Claudia\\Downloads\\trafic.csv'  
data = pd.read_csv(file_path)

minute = data['minut'].values
nr_masini = data['nr. masini'].values

print(minute)
print(nr_masini)

ore_creste = [7*60, 16*60]  
ore_descreste = [8*60, 19*60] 
intervale = [(4*60, 7*60), (7*60, 8*60), (8*60, 16*60), (16*60, 19*60), (19*60, 24*60)]

with pm.Model() as model:
    lam = pm.Gamma('lam', alpha=2, beta=1) 

    delta_creste = pm.Normal('delta_creste', mu=0, sd=2)
    delta_descreste = pm.Normal('delta_descreste', mu=0, sd=2)

    traffic_change = pm.Deterministic('traffic_change',
        (delta_creste * pm.math.sum(ore_creste[0] < minute)) +
        (delta_creste * pm.math.sum(ore_creste[1] < minute)) +
        (delta_descreste * pm.math.sum(ore_descreste[0] < minute)) +
        (delta_descreste * pm.math.sum(ore_descreste[1] < minute))
    )

    traffic = pm.Poisson('traffic', mu=lam + traffic_change, observed=nr_masini)

    trace = pm.sample(2000, tune=1000, cores=1) 
    # trace = pm.sample(20000)
    for i in range(len(intervale)):
        lambda_samples = trace['lam'] + trace['traffic_change'][:, i] 
        percentile_025 = np.percentile(lambda_samples, 2.5)  
        percentile_975 = np.percentile(lambda_samples, 97.5) 
        most_likely_lambda = np.mean(lambda_samples)  

        print(f"Intervalul {i+1} (de la {intervale[i][0]//60} la {intervale[i][1]//60} ore):")
        print(f"Capetele intervalului de încredere al lui λ: {percentile_025} - {percentile_975}")
        print(f"Cea mai probabilă valoare a lui λ: {most_likely_lambda}")

# pm.summary(trace)
# pm.traceplot(trace)
