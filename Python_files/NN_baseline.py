import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import torch.optim as optim
import scipy.io as sio
from matplotlib.lines import Line2D
from torch.utils.data import TensorDataset, DataLoader
import imageio  # For GIF creation

# (Optional workaround for OpenMP duplicate runtime)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is {}".format(device))


###############################################################################
# LSTM Network Definition
###############################################################################

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=50, output_dim=100):
        """
        LSTM network that processes a time sequence and outputs the predicted positions
        along the stick at each time step.

        Args:
            input_size (int): Number of features per time step (here, 1 for the time value).
            hidden_size (int): Hidden state size of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            output_dim (int): Number of spatial points per time step.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_length, hidden_size)
        output = self.fc(lstm_out)  # shape: (batch_size, seq_length, output_dim)
        return output


###############################################################################
# Helper Functions
###############################################################################

def generate_stick_swing_data(t_start=0, t_end=10, dt=0.005, N_points=101, L=1.0):
    """
    Simulates a swinging stick and returns time points and corresponding positions.

    Returns:
        t_eval: 1D numpy array of time steps.
        y_positions: 2D numpy array of shape (time_steps, N_points).
    """
    t_eval = np.arange(t_start, t_end, dt)
    theta_0 = np.pi / 4 + np.random.normal(0.3, 0.5)

    def theta_ode(t, y):
        g = 9.81
        L_val = 1.0
        theta, omega = y
        return [omega, -(g / L_val) * np.sin(theta)]

    y0 = [theta_0, 0.0]
    sol = solve_ivp(theta_ode, [t_start, t_end], y0, t_eval=t_eval)
    theta_values = sol.y[0]

    points_along_stick = np.linspace(0, L, N_points)
    y_positions = np.zeros((len(t_eval), N_points))
    for i, theta in enumerate(theta_values):
        y_positions[i, :] = points_along_stick * np.sin(theta)
    return t_eval, y_positions


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def test_nn(model, test_inputs):
    """
    Tests the LSTM model on new time inputs.
    If input is 2D (seq_length, input_size), adds a batch dimension.
    """
    model.eval()
    if isinstance(test_inputs, np.ndarray):
        inp = torch.tensor(test_inputs, dtype=torch.float32)
    else:
        inp = test_inputs
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    inp = inp.to(device)
    with torch.no_grad():
        predictions = model(inp).cpu().detach().numpy()
    return predictions


def conformal_prediction(Y_true=None, Y_pred=None, quant=0.1):
    """
    Computes the overall conformal loss as the 0.1 quantile of the average absolute error
    over all spatial points, computed across all samples and time steps.

    For each sample and time step, the average absolute error (across spatial points)
    is computed. These errors are flattened and the quantile is taken.
    """
    if Y_true is None or Y_pred is None:
        raise ValueError("Y_true and Y_pred must not be None")

    # Convert tensors to numpy arrays if needed
    if isinstance(Y_true, torch.Tensor):
        Y_true_np = Y_true.cpu().detach().numpy()
    else:
        Y_true_np = Y_true
    if isinstance(Y_pred, torch.Tensor):
        Y_pred_np = Y_pred.cpu().detach().numpy()
    else:
        Y_pred_np = Y_pred

    # Calculate average absolute error over spatial points (axis=2)
    errors = np.mean(np.abs(Y_true_np - Y_pred_np), axis=2)  # shape: (N_samples, time_steps)
    all_errors = errors.flatten()
    conf_loss = np.quantile(all_errors, quant)
    print("Conformal loss (quantile {:.2f}): {:.6f}".format(quant, conf_loss))
    return conf_loss


def train_model(model, dataloader, num_epochs=300, lr=0.0001):
    """
    Trains the LSTM model using MSE loss over data provided by the dataloader.
    """
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataloader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    return model


###############################################################################
# Main Experiment
###############################################################################

def run_experiment():
    # ------------------------- 1. Generate Synthetic Training Data -------------------------
    N_points = 101  # Number of spatial points (should match test data)
    L = 1.0
    t_start, t_end, dt = 0, 10, 0.005  # Determine training sequence length from dt
    time_steps = int((t_end - t_start) / dt)

    N_samples = 1000  # Number of synthetic samples for training
    t_samples = []
    y_samples = []
    for _ in range(N_samples):
        t_eval, y_positions = generate_stick_swing_data(t_start, t_end, dt, N_points, L)
        # Normalize time and reshape for LSTM input (each time step has one feature)
        t_eval_norm = t_eval.reshape(-1, 1)  # shape: (time_steps, 1)
        y_positions_norm = normalize(y_positions)  # shape: (time_steps, N_points)
        t_samples.append(t_eval_norm)
        y_samples.append(y_positions_norm)

    # Convert lists to numpy arrays and then to torch tensors
    t_samples_np = np.array(t_samples)  # shape: (N_samples, time_steps, 1)
    y_samples_np = np.array(y_samples)  # shape: (N_samples, time_steps, N_points)

    time_train_tensor = torch.tensor(t_samples_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_samples_np, dtype=torch.float32)

    # Create a TensorDataset and DataLoader for training
    train_dataset = TensorDataset(time_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # ------------------------- 2. Train the LSTM Model -------------------------
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, output_dim=N_points).to(device)
    print("Training on synthetic data using DataLoader ...")
    model = train_model(model, train_loader, num_epochs=300, lr=0.0001)

    # ------------------ 3. Compute Conformal Score on Generated Data ------------------
    # Get predictions on the entire training set
    model.eval()
    with torch.no_grad():
        train_preds = model(time_train_tensor.to(device)).cpu().detach()
    conformal_score = conformal_prediction(Y_true=Y_train_tensor, Y_pred=train_preds, quant=0.1)

    # ------------------ 4. Test on New Data (.mat file) ------------------
    # Load test data from the .mat file
    # TODO Adjust path as needed
    file_path = r'C:\Users\Desktop\PnP-PIML\video\Data\Aligned_center_points.mat'
    mat_contents = sio.loadmat(file_path)
    x = mat_contents['x'].flatten('F')
    y = mat_contents['y'].flatten('F')
    z = mat_contents['z'].flatten('F')

    # Reshape test data: assume z contains time and x, y contain spatial coordinates
    time_steps_test = np.unique(z)  # Use the original test time steps (do not resample)
    n_points_test = len(x) // len(time_steps_test)
    x_reshaped = x.reshape(len(time_steps_test), n_points_test)
    y_reshaped = y.reshape(len(time_steps_test), n_points_test)

    # Normalize test time and outputs (using min/max from the test data itself)
    test_time = normalize(time_steps_test)  # shape: (T_test,)
    y_test = normalize(y_reshaped)  # shape: (T_test, n_points_test)

    # Prepare test input tensor for the LSTM; note that the sequence length may differ
    test_time_tensor = torch.tensor(test_time.reshape(-1, 1), dtype=torch.float32).unsqueeze(0).to(device)

    test_predictions = test_nn(model, test_time_tensor)  # shape: (1, T_test, N_points)
    test_predictions = test_predictions[0]  # Remove batch dimension → shape: (T_test, N_points)

    # ------------------ 5. Compute Test MSE ------------------
    # Calculate the MSE for each time step by averaging the squared error over spatial points
    test_time_mse = np.mean((test_predictions - y_test) ** 2, axis=1)  # shape: (T_test,)
    overall_avg_test_mse = np.mean(test_time_mse)
    print("Overall average test MSE on .mat data: {:.6f}".format(overall_avg_test_mse))

    # ------------------ 6. Compare Performance ------------------
    # Plot the test time MSE curve and overlay the conformal score as a horizontal line
    plt.figure(figsize=(10, 6))
    plt.plot(test_time, test_time_mse, label='Test Time MSE (avg over spatial points)')
    plt.axhline(conformal_score, color='red', linestyle='--',
                label=f'Conformal Score (0.1 quantile on generated data): {conformal_score:.6f}')
    plt.xlabel('Normalized Time')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Test Time MSE vs. Conformal Score')
    plt.legend()
    plt.savefig('test_time_mse_vs_conformal_score.png')
    plt.show()

    # ------------------ 7. Visualize Test Results as a GIF ------------------
    # Here we generate a GIF that compares the LSTM prediction vs. the ground truth (GT)
    frames = []
    # For the GIF, we sample 100 frames from the test sequence.
    num_frames = 100
    indices = np.linspace(0, len(test_time)-1, num_frames, dtype=int)
    # Create a spatial axis for plotting (assumed normalized over [0, 1])
    spatial_axis = np.linspace(0, 1, y_test.shape[1])
    for i in indices:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(spatial_axis, y_test[i, :], label='Ground Truth', color='blue')
        ax.plot(spatial_axis, test_predictions[i, :], label='LSTM Prediction', color='red', linestyle='--')
        ax.set_title(f'Time step: {test_time[i]:.3f}')
        ax.set_xlabel('Spatial Position (normalized)')
        ax.set_ylabel('Stick Position (normalized)')
        ax.legend()
        # Render the figure and store as an image frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    gif_path = 'test_results_comparison.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"GIF saved to {gif_path}")

    return model


###############################################################################
# Run the Experiment
###############################################################################

if __name__ == "__main__":
    trained_model = run_experiment()
