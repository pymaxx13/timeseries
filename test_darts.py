import sys
import inspect
from darts.models import NeuralForecastModel
from engressionts.models.neuralforecast.enpatchtst import EnPatchTST
from neuralforecast.models.patchtst import PatchTST

model_kwargs_std = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_steps": 1,
    "random_seed": 42,
    "windows_batch_size": 1024,
}

model_kwargs_en = dict(model_kwargs_std)
model_kwargs_en.update({
    "num_samples_train": 2,
    "noise_std": 1.0,
    "noise_type": "uniform",
})

model1 = NeuralForecastModel(
    input_chunk_length=24,
    output_chunk_length=24,
    model=PatchTST,
    model_kwargs=model_kwargs_std,
    pl_trainer_kwargs={"accelerator": "cpu"},
)
print("PatchTST nf_model_params:", model1.nf_model_params)

model2 = NeuralForecastModel(
    input_chunk_length=24,
    output_chunk_length=24,
    model=EnPatchTST,
    model_kwargs=model_kwargs_en,
    pl_trainer_kwargs={"accelerator": "cpu"},
)
print("EnPatchTST nf_model_params:", model2.nf_model_params)

print("\nDoes EnPatchTST signature have max_steps?")
sig = inspect.signature(EnPatchTST.__init__)
print("max_steps" in sig.parameters)
