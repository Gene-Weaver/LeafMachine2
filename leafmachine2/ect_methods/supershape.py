import numpy as np
import os, torch
from tqdm import tqdm

class Supershape:
    def __init__(self, m, n1, n2, n3, device='cpu'):
        self.m = m
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.a = 1  # Can be parameterized if needed
        self.b = 1  # Can be parameterized if needed
        self.device = device  # Device to run the calculations on (CPU or GPU)

    # def evaluate(self, phi):
    #     """
    #     Evaluate the supershape function for a given angle `phi`.
    #     Returns (x, y) coordinates, with clamping to ensure valid values.
    #     """
    #     # Calculate r based on the supershape formula
    #     t1 = np.cos(self.m * phi / 4) / self.a
    #     t1 = np.abs(t1) ** self.n2

    #     t2 = np.sin(self.m * phi / 4) / self.b
    #     t2 = np.abs(t2) ** self.n3

    #     # Clamp r to avoid extremely large or small values
    #     r = np.clip((t1 + t2) ** (-1 / self.n1), 1e-10, 1e10)
        
    #     # Calculate x and y based on r and phi
    #     x = r * np.cos(phi)
    #     y = r * np.sin(phi)

    #     # Clamp x and y to avoid excessively large values (optional, adjust limits as needed)
    #     x = np.clip(x, -1e10, 1e10)
    #     y = np.clip(y, -1e10, 1e10)

    #     return x, y
    def evaluate(self, phi_tensor):
        """
        Evaluate the supershape function for a batch of angles `phi` using PyTorch tensors.
        Returns (x, y) tensors with clamping to ensure valid values.
        """
        # Calculate t1 and t2 based on the supershape formula
        t1 = torch.cos(self.m * phi_tensor / 4) / self.a
        t1 = torch.abs(t1) ** self.n2

        t2 = torch.sin(self.m * phi_tensor / 4) / self.b
        t2 = torch.abs(t2) ** self.n3

        # Calculate r with clamping
        r = torch.clamp((t1 + t2) ** (-1 / self.n1), 1e-10, 1e10)

        # Calculate x and y based on r and phi
        x = r * torch.cos(phi_tensor)
        y = r * torch.sin(phi_tensor)

        # Clamp x and y to avoid excessively large values
        x = torch.clamp(x, -1e10, 1e10)
        y = torch.clamp(y, -1e10, 1e10)

        return x, y

def save_contour_to_file(filename, x_vals, y_vals):
    """
    Save the (x, y) contour points to a text file.
    The first 11 rows will be padded with `-999` as a header,
    and any unused rows will also be padded with `-999`.
    """
    num_points = len(x_vals)
    with open(filename, 'w') as file:
        # Write the first 11 rows as -999,-999 (header placeholder)
        for _ in range(11):
            file.write("-999,-999\n")
        
        # Write the actual contour points
        for i in range(num_points):  # Adjusting for the 11 header lines
            file.write(f"{x_vals[i]},{y_vals[i]}\n")

def parameter_sweep(m_vals, n1_vals, n2_vals, n3_vals, output_dir, num_points=100, device='cpu'):
    """
    Perform a parameter sweep for the given lists of m, n1, n2, and n3 values.
    Save the contours (x, y) to text files with each row representing an (x, y) coordinate.
    """
    # phi_vals = np.linspace(0, 2 * np.pi, num_points)
    phi_tensor = torch.linspace(0, 2 * torch.pi, num_points, device=device)


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for m in tqdm(m_vals, desc="Processing m values"):
        for n1 in n1_vals:
            for n2 in n2_vals:
                for n3 in n3_vals:
                    # supershape = Supershape(m, n1, n2, n3)
                    # x_vals, y_vals = [], []

                    # # Evaluate (x, y) for each phi
                    # for phi in phi_vals:
                    #     x, y = supershape.evaluate(phi)
                    #     x_vals.append(x)
                    #     y_vals.append(y)

                    # # Construct a filename based on the parameters
                    # m_str = str(m).replace('.', '-')
                    # n1_str = str(n1).replace('.', '-')
                    # n2_str = str(n2).replace('.', '-')
                    # n3_str = str(n3).replace('.', '-')

                    # # Construct a filename based on the parameters
                    # filename = f"supershape_m{m_str}_n1{n1_str}_n2{n2_str}_n3{n3_str}.txt"
                    
                    # file_path = os.path.join(output_dir, filename)

                    # # Save the contour points to the text file
                    # save_contour_to_file(file_path, x_vals, y_vals)
                    # # print(f"Contour saved to {file_path}")
                    # Initialize the supershape with the current parameters
                    supershape = Supershape(m, n1, n2, n3, device=device)

                    # Use PyTorch to evaluate all (x, y) values for the given phi_tensor
                    x_vals, y_vals = supershape.evaluate(phi_tensor)

                    # Move tensors back to CPU and convert to lists for saving
                    x_vals = x_vals.cpu().numpy().tolist()
                    y_vals = y_vals.cpu().numpy().tolist()

                    # Construct a filename based on the parameters
                    m_str = str(m).replace('.', '-')
                    n1_str = str(n1).replace('.', '-')
                    n2_str = str(n2).replace('.', '-')
                    n3_str = str(n3).replace('.', '-')

                    # Construct a filename based on the parameters
                    filename = f"supershape_m{m_str}_n1{n1_str}_n2{n2_str}_n3{n3_str}.txt"
                    file_path = os.path.join(output_dir, filename)

                    # Save the contour points to the text file
                    save_contour_to_file(file_path, x_vals, y_vals)


if __name__ == "__main__":
    # Parameter sweep lists based on Bourke's descriptions
    # m_vals = [0, 1, 3, 6, 1/6, 7/6, 19/6]   # Symmetry factor
    # n1_vals = [0.1, 0.3, 0.5, 1, 10, 40, 60]  # General shape control
    # n2_vals = [0.1,  5, 10, 55]  # Pinching and symmetry
    # n3_vals = [0.1,  5, 10, 55]  # Similar to n2

    m_first = np.arange(0.1, 1.05, 0.05)
    m_second = np.linspace(1, 10, 10)
    m_vals = np.concatenate((m_first, m_second))


    # Create the first linspace from 0.1 to 1 with 10 values
    n_first = np.arange(0.1, 1.05, 0.05)
    # Create the second linspace from 1 to 100 with 20 values
    n_second = np.arange(5, 105, 5)
    n1_vals = np.concatenate((n_first, n_second))

    n2_vals = n1_vals
    n3_vals = n1_vals

    # Define the output directory
    output_dir = 'D:/D_Desktop/LM2_Cornales/experiments/supershapes_sweep'  

    # Choose the device: use GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Perform the parameter sweep and plot the results
    parameter_sweep(m_vals, n1_vals, n2_vals, n3_vals, output_dir, device=device)
