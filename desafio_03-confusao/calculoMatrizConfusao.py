import matplotlib.pyplot as plt

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) != 0 else 0

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r) if (p + r) != 0 else 0

# Exemplo de uso
tp, tn, fp, fn = 50, 100, 10, 40
acc = accuracy(tp, tn, fp, fn)
rec = recall(tp, fn)
spec = specificity(tn, fp)
prec = precision(tp, fp)
f1 = f1_score(tp, fp, fn)

print("Acurácia:", acc)
print("Sensibilidade:", rec)
print("Especificidade:", spec)
print("Precisão:", prec)
print("F1-score:", f1)

# Criando gráfico das métricas
metrics = ['Acurácia', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1-score']
values = [acc, rec, spec, prec, f1]

plt.figure(figsize=(8,5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.ylim(0, 1)
plt.xlabel("Métricas")
plt.ylabel("Valores")
plt.title("Desempenho do Modelo de Classificação")
plt.show()
