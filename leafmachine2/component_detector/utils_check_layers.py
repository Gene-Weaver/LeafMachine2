import torch
import torch.nn as nn
from torchsummary import summary

# Load the model
model   = torch.load('D:/Dropbox/FieldPrism/fieldprism/yolov5/weights_nano/best.pt')

summary(model['model'] , input_size=(3, 512, 512))

model.load_state_dict(checkpoint['model'])
# Create a dummy input with the same dimensions expected by the model. 
# For a YOLO model, it might be something like (batch_size, 3, height, width)
dummy_input = torch.randn(1, 3, 512, 512)

# Get a prediction to inspect the shape
with torch.no_grad():
    output = model(dummy_input)

# Print the output shape
print("Output shape:", output.shape)