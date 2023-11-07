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

- cu cat Y este mai mare cu atat estimarea nr total de clienti este mai buna (mai multe observatii => distributia se va concentra in jurul valorii mai mari)
- daca Y este mic distributia pentru poate avea o medie mai mica (mai putini vizitatori)
- daca theta este mare, distributia pentru n reflecta o estimare mai mare a nr de clienti ce pot cumpara
- daca theta este mic ar putea duce la o distributie pentru n cu medie mai mica (se asteapta o frecventa mai mica a clientilor)
