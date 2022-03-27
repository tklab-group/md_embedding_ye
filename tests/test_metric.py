from model.metric import Metric
m = Metric()
aq = ["a", "b", "c"]
eq = ["c"]
ap = m.ap(3, aq, eq)
print(ap)
print('precision k', m.pk(aq, eq, 3))
print('recall k', m.rk(aq, eq, 3))
