from functools import partial

import bitsandbytes as bnb
import torch


def quantize_layer(module, quantization_type="8bit"):
    if quantization_type == "16bit":
        QuantizedLinearLayer = partial(torch.nn.Linear, dtype=torch.bfloat16)
    elif quantization_type == "8bit":
        QuantizedLinearLayer = bnb.nn.Linear8bitLt
    elif quantization_type == "4bit":
        QuantizedLinearLayer = bnb.nn.Linear4bit
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

    # QuantizedLinearLayer = bnb.nn.Linear8bitLt if quantization_type == "8bit" else bnb.nn.Linear4bit
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            layer = QuantizedLinearLayer(child.in_features, child.out_features, bias=child.bias is not None)
            layer.load_state_dict(child.state_dict())
            setattr(module, name, layer)
        else:
            # Recursively apply to children
            quantize_layer(child, quantization_type)
    return module


# TODO: bias
def dequantize_layer(module):
    for name, child in module.named_children():
        # Check if the child is a 4-bit quantized layer
        if isinstance(child, bnb.nn.Linear4bit) or isinstance(child, bnb.nn.Linear8bitLt):
            # Create a new torch.nn.Linear layer with the same dimensions
            dequantized_layer = torch.nn.Linear(
                in_features=child.in_features, out_features=child.out_features, bias=child.bias is not None
            )

            dequantized_weight = bnb.functional.dequantize_4bit(child.weight, child.weight.quant_state)
            # Load the weights and biases from the quantized layer into the new layer
            dequantized_layer.weight = torch.nn.Parameter(dequantized_weight.detach().to(torch.float32))

            # Replace the quantized layer with the new dequantized layer
            setattr(module, name, dequantized_layer)
        else:
            # Recursively apply to children
            dequantize_layer(child)
    return module
