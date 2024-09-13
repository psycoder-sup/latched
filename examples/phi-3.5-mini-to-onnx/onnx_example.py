# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnxruntime as ort

from transformers import AutoTokenizer, AutoModelForCausalLM

from latched.model_wrappers.auto import AutoModelWrapper
from latched.model_optimizers.auto import AutoOptimizer
from latched.model_exporters.onnx import ONNXExporter

# Load the huggingface tokenizer and model
model_path = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

# Wrap the model in Latched's AutoModel
latched_model_wrapper = AutoModelWrapper(model=model, tokenizer=tokenizer).create()

# Optimize the model
optimized_model_wrapper = AutoOptimizer.run(latched_model_wrapper)

# Export the model to ONNX
# For the many cases, you can use AutoExporter to automatically select the best exporter for the given model.
# However, for this specific case, we can use ONNXExporter directly.
ONNXExporter.run(optimized_model_wrapper, output_name="phi-3.5-mini-instruct.onnx")

# Test the onnx model
ort_session = ort.InferenceSession("phi-3.5-mini-instruct.onnx")
dummy_input = tokenizer("Hello, world!", return_tensors="pt")
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input["input_ids"]}
ort_outputs = ort_session.run(None, ort_inputs)
# TODO: ONNXruntime doesn't work yet. Need to fix the model.
print(ort_outputs)
