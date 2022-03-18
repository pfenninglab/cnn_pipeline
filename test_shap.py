from train import CNN
from dataset import FaExampleIterator
import torch

ds = FaExampleIterator(part='train', label='pos')
for x, _ in ds:
	break

t_model = CNN()
example = torch.Tensor(x).unsqueeze(0)
torch.onnx.export(t_model, example, "../../models/onnx/model_2d.onnx", verbose=True,
	input_names = ['input'], output_names = ['output'])




from train import CNN
from dataset import FaExampleIterator

from shap import DeepExplainer
import torch
import numpy as np

from itertools import islice


bg_data = np.array([x for x, _ in islice(FaExampleIterator('neg', 'val'), 10)])
fg_data = np.array([x for x, _ in islice(FaExampleIterator('pos', 'val'), 10)])

import onnx
from onnx2keras import onnx_to_keras

print("loading onnx model")
onnx_model = onnx.load('../../models/onnx/model_2d.onnx')
k_model = onnx_to_keras(onnx_model, ['input'])
print("loading onnx model... done")
print(k_model.summary())


print("check equal:")
t_model.eval()
res_t = t_model(torch.Tensor(bg_data))
res_k = k_model.predict(bg_data)

print((torch.Tensor(res_k) - res_t).abs())



print("DeepSHAP on Keras model:")
e = DeepExplainer(k_model, bg_data)
res = e.shap_values(fg_data)

print(res)
