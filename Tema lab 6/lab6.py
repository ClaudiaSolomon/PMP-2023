import pymc3 as pm
import arviz as az 

#1
Y = [0, 5, 10]
Theta = [0.2, 0.5]
if __name__ == '__main__':
    with pm.Model() as model:
        n = pm.Poisson('n', mu=10)
        for y in Y:
            for theta in Theta:
                nr_clienti = pm.Binomial(f'Nr clienti pentru={y} si theta={theta}', n=n, p=theta, observed=y)
        trace = pm.sample(2000, tune=1000, chains=2)
        az.plot_posterior(trace)


#2
- cu cat Y creste cu atat distributia lui n se modifica deoarece ea reprezinta frecventa cumpararilor a unui anumit nr de clienti
- theta afecteaza estimarea nr total de clienti