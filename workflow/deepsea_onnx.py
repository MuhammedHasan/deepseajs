import scipy
import torch
import onnxruntime as ort
from .deepsea import DeepSeaGradCam

mat = scipy.io.loadmat(snakemake.input['test'])
X = torch.FloatTensor(mat['testxdata']).to('cpu')
y = torch.FloatTensor(mat['testdata']).to('cpu')

output_size = y.shape[1]
model = DeepSeaGradCam(output_size=output_size)

state_dict = torch.load(snakemake.input['model'])
model.load_state_dict(state_dict)

torch.onnx.dynamo_export(model, X[:1]).save(snakemake.output['onnx'])

ort_sess_gradcam = ort.InferenceSession(snakemake.output['onnx'])
output = ort_sess_gradcam.run(None, {'arg0': X[:1].numpy()})

print(output[0].shape)
print(output[1].shape)