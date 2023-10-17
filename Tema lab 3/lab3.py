from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

#I=incendiu, C=cutremur, A=alarma

model = BayesianNetwork([('C', 'I'), ('C', 'A'),('I','A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) 

cpd_i = TabularCPD(variable='I', variable_card=2, 
                   values=[[0.99, 0.03],
                           [0.01, 0.97]],
                   evidence=['C'],
                   evidence_card=[2])

cpd_a = TabularCPD(variable='A', variable_card=2, 
                    values=[[0.9999, 0.0001, 0.02, 0.95],
                            [0.0001, 0.9999, 0.98, 0.05]],
                    evidence=['C', 'I'],
                    evidence_card=[2, 2])

model.add_cpds(cpd_c, cpd_i, cpd_a)
assert model.check_model()

#b
inference = VariableElimination(model)
probability = inference.query(variables=['C'], evidence={'A': 1})
print(probability)

#c
probability = inference.query(variables=['I'], evidence={'A': 0})
print(probability)