import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

#a
file_path = 'C:\\Users\\Claudia\\Documents\\GitHub\\PMP-2023\\Examen\\BostonHousing.csv'  
data = pd.read_csv(file_path)
medv=data['medv'].values
rm=data['rm'].values
crim=data['crim'].values
indus=data['indus'].values

#b
#modelul e de forma niu=alfa+beta1*x1+beta2*x2+beta3*x3
#unde y~N(niu,sigma),y=medv,x1=rm,x2=crim,x3=indus
if __name__ == '__main__':
    with pm.Model() as model_locuinte:       
            alfa = pm.Normal('alfa', mu=0, sigma=10)
            beta1 = pm.Normal('beta1', mu=0, sigma=1)
            beta2 = pm.Normal('beta2', mu=0, sigma=1)
            beta3 = pm.Normal('beta3', mu=0, sigma=1)
            sigma = pm.HalfCauchy('sigma', 5)

            niu = pm.Deterministic('niu', alfa + beta1 * rm + beta2 * crim+beta3*indus)
            y_pred = pm.Normal('y_pred', mu=niu, sigma=sigma, observed=medv)

            idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta1', 'beta2','beta3', 'sigma'])
    plt.show()

#c
    az.plot_forest(idata,hdi_prob=0.95,var_names=['beta1','beta2','beta3'])
    summary=az.summary(idata,hdi_prob=0.95,var_names=['beta1','beta2','beta3'])
    plt.show()

    #interval incredere 2.5%-97.5%
    is_beta1_significant = (summary['hdi_2.5%']['beta1'] > 0) or (summary['hdi_97.5%']['beta1'] < 0)
    is_beta2_significant = (summary['hdi_2.5%']['beta2'] > 0) or (summary['hdi_97.5%']['beta2'] < 0)
    is_beta3_significant = (summary['hdi_2.5%']['beta3'] > 0) or (summary['hdi_97.5%']['beta3'] < 0)

    if(is_beta1_significant):
          print("Nr mediu de camere este un predictor util")
    else:
          print("Nr mediu de camere nu este un predictor util")
    if(is_beta2_significant):
          print("Rata criminalitatii este un predictor util")
    else:
          print("Rata criminalitatii nu este un predictor util")
    if(is_beta3_significant):
          print("Proportia suprafetei comerciale non-retail este un predictor util")
    else:
          print("Proportia suprafetei comerciale non-retail nu este un predictor util")
    
    #din figura se observa ca x2 si x3 au o mica influenta asupra modelului (crim si indus)->beta2 si beta3 sunt mici in comparatie cu beta1;
    #cea mai mare influenta o are x1(rm)
    #ordinea de influenta ar fi: x1,x3,x2

#d
    ppc = pm.sample_posterior_predictive(idata, model=model_locuinte)
    posterior_predictive = ppc['y_pred']

    hdi_50_posterior_predictive = az.hdi(posterior_predictive.flatten(), hdi_prob=0.5)

    print(f"Intervalul de 50% HDI pentru distribuția predictivă posterioară este: "
          f"[{hdi_50_posterior_predictive[0]:.2f}, {hdi_50_posterior_predictive[1]:.2f}]")

   