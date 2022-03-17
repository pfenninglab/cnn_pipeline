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
onnx_model = onnx.load('/home/csestili/models/onnx/model.onnx')
k_model = onnx_to_keras(onnx_model, ['input'])
print("loading onnx model... done")
print(k_model.summary)

print("loading pytorch model")
t_model = CNN.load_from_checkpoint('lightning_logs/version_2608479/checkpoints/epoch=79-step=137519.ckpt')
from torchsummary import summary
print(summary(t_model, bg_data[0].size))
print("loading pytorch model... done")


print("check equal:")
res_t = t_model(torch.Tensor(bg_data))
res_k = k_model.predict(bg_data)

print((torch.Tensor(res_k) - res_t).abs())



print("DeepSHAP on Keras model:")
e = DeepExplainer(k_model, bg_data)
res = e.shap_values(fg_data)

print(res)
