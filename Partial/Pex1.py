import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from pgmpy.inference import VariableElimination


# stema_monedaJ0 = stats.binom.rvs(1,0.5, size=10)
# stema_monedaJ1 = stats.binom.rvs(1,0.6, size=10) #2/3 sa obtine stema
#1
castig_J0=0
castig_J1=1
joc=10000
while joc:
    moneda_normala=stats.binom.rvs(1,0.5,size=1) #se arunca o data pt inceput

    if moneda_normala[0]==0:
        stema_monedaJ0 = stats.binom.rvs(1,0.5, size=1)
        n=stema_monedaJ0[0]
        stema_monedaJ1 = stats.binom.rvs(1,0.6, size=n+1)
        m=0
        for j in range(n+1):
            if stema_monedaJ1[j]==1:
                m+=1
    else:
        stema_monedaJ1 = stats.binom.rvs(1,0.6, size=1)
        n=stema_monedaJ1[0]
        stema_monedaJ0 = stats.binom.rvs(1,0.5, size=n+1)
        m=0
        for j in range(n+1):
            if stema_monedaJ0[j]==1:
                m+=1
    if n>=m:
        castig_J0+=1
    else:
        castig_J1+=1

    joc-=1

if castig_J0>castig_J1:
    print("Castiga J0 mai mult")
else:
    print("Castiga J1 mai mult")

#2
joc_model = BayesianNetwork([('M', 'J0'), ('M', 'J1'), ('J0', 'C'),
('J1', 'C')])
#M moneda initiala de incepere
#C cine castiga jocul
pos = nx.circular_layout(joc_model)
nx.draw(joc_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
CPD_M = TabularCPD(variable='M', variable_card=2, values=[[0.5], [0.5]])
CPD_J0 = TabularCPD(variable='J0', variable_card=2,
                   values=[[0.5, 0.5],
                          [0.5, 0.5]],
                   evidence=['M'],
                  evidence_card=[2])
CPD_J1 = TabularCPD(variable='J1', variable_card=2,
                   values=[[0.6, 0.6],
                          [0.4, 0.4]],
                   evidence=['M'],
                  evidence_card=[2])
CPD_C = TabularCPD(variable='C', variable_card=3,
values=[[0.5, 0.7, 0.02, 0.2],
        [0.25, 0.25, 0.08, 0.3],
        [0.25, 0.05, 0.9, 0.5]],
evidence=['J0', 'J1'],
evidence_card=[2, 2])
joc_model.add_cpds(CPD_M, CPD_J0, CPD_J1,CPD_C)
assert joc_model.check_model()

#3
#daca s-a obtnut stema o data runda 2=> in prima runda se pot obtine 0 steme,deci J1 castiga
#                                    => sau in prima runda s-a obtinut >=1 steme, deci J0 castiga
infer = VariableElimination(joc_model)
posterior_p_1 = infer.query(["M"], evidence={"J1": 1,"J0":0})
posterior_p_2 = infer.query(["M"], evidence={"J1": 0,"J0":1})

print(posterior_p_1+posterior_p_2)

