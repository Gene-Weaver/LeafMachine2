import torch
import torch.jit
import sys
import os

def check_if_jit_model(model_path):
    try:
        # Try to load the model as a TorchScript model
        torch.jit.load(model_path)
        return True, "The model is in TorchScript format."
    except Exception as e:
        return False, f"The model is not in TorchScript format. Error: {str(e)}"

# Append the path to your system path (Note: Don't use quotes)
sys.path.append("D:/Dropbox/FieldPrism/fieldprism/yolov5/weights")


# Check if the path exists and you have read permissions
model_path = "D:/Dropbox/FieldPrism/fieldprism/yolov5/weights_nano/best.pt"


if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist. Please check the path.")
    sys.exit(1)


is_jit_model, message = check_if_jit_model(model_path)
print(message)


# Load your custom model
# Load your custom model
try:
    loaded_dict = torch.load(model_path, map_location='cuda:0')  # Adjust the device as needed

    # Assuming your model architecture is defined in a class called `YourModelClass`
    # model = YourModelClass() 
    # model.load_state_dict(loaded_dict['model'])  # If the model is saved as a state dictionary

    # If the model is saved entirely (architecture + weights)
    if isinstance(loaded_dict, dict) and 'model' in loaded_dict:
        model = loaded_dict['model']
    else:
        model = loaded_dict  # Assuming the loaded object is a model
    
    # Switch the model to evaluation mode
    model.eval()

    # Create a dummy input that matches the input dimensions of the model
    dummy_input = torch.randn(1, 3, 512, 512).half().to('cuda:0')


    # Try tracing the model
    try:
        scripted_module = torch.jit.trace(model, dummy_input)
        print("Model traced successfully.")
    except Exception as e:
        print(f"An error occurred during tracing. Error: {str(e)}")

    # Try scripting the model
    try:
        scripted_module = torch.jit.script(model)
        save_path = "D:/Dropbox/FieldPrism/fieldprism/yolov5/weights/fieldprism_v_1_0.pt"
        scripted_module.save(save_path)
        print(f"Saved TorchScript model to {save_path}")
    except Exception as e:
        print(f"An error occurred during the scripting. Error: {str(e)}")
except Exception as e:
    print(f"Error in loading the model: {e}")
    sys.exit(1)


# Script the model
try:
    scripted_module = torch.jit.script(model)
    # Save the TorchScript model (make sure you have write permissions for the directory)
    save_path = "D:/Dropbox/FieldPrism/fieldprism/yolov5/weights/fieldprism_v_1_0.pt"
    scripted_module.save(save_path)
    print(f"Saved TorchScript model to {save_path}")
except Exception as e:
    print(f"An error occurred during the scripting. Error: {str(e)}")
