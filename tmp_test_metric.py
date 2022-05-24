from metrics import Specificity, MulticlassMetric
from tensorflow.keras.metrics import Precision
import numpy as np


y_true = np.array([1, 1, 0, 0, 0, 0])
y_true2 = np.array([1, 1, 1, 0, 0, 0])
y_pred = np.array([0.9, 0, 0.9, 0, 0.7, 0.8])

m = Specificity()
m.update_state(y_true, y_pred)
m.update_state(y_true2, y_pred)
# tn / (tn + fp)
print(float(m.result()))

y_pred = np.array([
	[0.1, 0.9],
	[1.0, 0.0],
	[0.1, 0.9],
	[1.0, 0.0],
	[0.3, 0.7],
	[0.2, 0.8]
])
m = MulticlassMetric('Specificity', name='specificity', pos_label=1)
m.update_state(y_true, y_pred)
m.update_state(y_true2, y_pred)
print(float(m.result()))