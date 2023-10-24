import numpy as np
import scipy.stats as stats

#1
lambda_ = 20

medie = 2
deviatie_standard = 0.5

alpha = 1.0

numar_clienti = np.random.poisson(lambda_)
plasare_comanda = np.random.normal(medie, deviatie_standard, numar_clienti)
pregatire_comanda = np.random.exponential(alpha, numar_clienti)

timp_servire_total = sum(plasare_comanda) + sum(pregatire_comanda)

print("Numărul de clienți într-o oră:", numar_clienti)
print("Timpul total de servire a clienților:", timp_servire_total)

#2
probabilitate = 0.95
timp = 15
alpha = 1000
while True:
    service_time = np.random.exponential(alpha, numar_clienti).sum()
    probability = 1 - stats.expon.cdf(timp, scale=1/alpha)
    if probability >= probabilitate:
        break
    alpha -= 0.01 

print("Valoarea maximă a lui alpha:", alpha)

#3
pregatire_comanda = np.random.exponential(alpha, numar_clienti)
timp_servire_total = sum(plasare_comanda) + sum(pregatire_comanda)

timp_servire_mediu = timp_servire_total / numar_clienti

print("Timpul mediu de așteptare:", timp_servire_mediu)