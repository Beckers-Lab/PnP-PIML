import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
import imageio
import pandas as pd
from matplotlib.patches import Patch

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("The device is {}".format(device))


# ---------------------------
# Helper Functions
# ---------------------------
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def generate_stick_swing_data(t_start=0, t_end=10, dt=0.005, N_points=101, L=1.0):
    """
    Simulates a swinging stick and returns time points and corresponding positions.
    """
    t_eval = np.arange(t_start, t_end, dt)
    theta_0 = np.pi / 4 + np.random.normal(0.3, 0.5)

    def theta_ode(t, y):
        g = 9.81
        L_val = L
        theta, omega = y
        return [omega, -(g / L_val) * np.sin(theta)]

    y0 = [theta_0, 0.0]
    from scipy.integrate import solve_ivp
    sol = solve_ivp(theta_ode, [t_start, t_end], y0, t_eval=t_eval)
    theta_values = sol.y[0]
    points_along_stick = np.linspace(0, L, N_points)
    y_positions = np.zeros((len(t_eval), N_points))
    for i, theta in enumerate(theta_values):
        y_positions[i, :] = points_along_stick * np.sin(theta)
    return t_eval, y_positions


def create_sliding_windows(data, input_seq_len, output_seq_len, stride=10):
    """
    Creates sliding window examples from a sequence.
    Args:
        data: numpy array of shape (time_steps, n_points)
        input_seq_len: number of consecutive frames used as input.
        output_seq_len: number of consecutive frames to predict.
        stride: step size between windows.
    Returns:
        X: array of shape (num_windows, input_seq_len, n_points)
        Y: array of shape (num_windows, output_seq_len, n_points)
    """
    X, Y = [], []
    total = data.shape[0]
    for i in range(0, total - input_seq_len - output_seq_len + 1, stride):
        X.append(data[i: i + input_seq_len])
        Y.append(data[i + input_seq_len: i + input_seq_len + output_seq_len])
    return np.array(X), np.array(Y)


# ---------------------------
# Conformal Prediction & Error Metrics
# ---------------------------
def compute_cp_bound(Y_true, Y_pred, alpha=0.1):
    """
    Computes the CP bound for a 2D array of errors (N_windows, n_points).
    For each sample, compute the average absolute error over spatial points,
    then sort all these errors in descending order and select the error at rank
    ceil(alpha * N) - 1.

    Args:
        Y_true, Y_pred: arrays of shape (N_windows, n_points)
        alpha: fraction (e.g., 0.1 for the top 10% highest error)
    Returns:
        cp_bound: the CP bound computed as described,
        errors: the array of average errors for each window.
    """
    # Ensure inputs are numpy arrays.
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().detach().numpy()
    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.cpu().detach().numpy()

    # Compute the average absolute error over spatial points for each window.
    squared_diff = np.square(Y_pred - Y_true)
    # Average over the outputs and spatial dimensions for each time step (axes 1 and 2)
    errors = np.mean(squared_diff, axis=1)

    # Sort errors in descending order (largest error first)
    sorted_errors = np.sort(errors)[::-1]

    # Determine the index for the top alpha fraction.
    idx = max(int(np.ceil(alpha * len(sorted_errors))) - 1, 0)
    cp_bound = sorted_errors[idx]
    print(cp_bound)

    return cp_bound


def compute_mse_over_time(Y_true, Y_pred):
    """
    Computes the mean squared error (MSE) averaged over spatial points.
    If the inputs are 3D (N_windows, output_seq_len, n_points), it computes a separate MSE per time step.
    If the inputs are 2D (N_windows, n_points), it computes a single MSE.

    Returns:
        mse_over_time: If 3D input, an array of length output_seq_len; if 2D input, a one-element array.
    """
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().detach().numpy()
    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.cpu().detach().numpy()

    if isinstance(Y_true, list):
        Y_true = np.array(Y_true)
    if isinstance(Y_pred, list):
        Y_pred = np.array(Y_pred)

    squared_diff = np.square(Y_pred - Y_true)
    # Average over the outputs and spatial dimensions for each time step (axes 1 and 2)
    mse_per_time = np.mean(squared_diff, axis=(1, 2))
    return mse_per_time


# ---------------------------
# Sequence-to-Sequence LSTM Model (Encoder-Decoder)
# ---------------------------
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_points, hidden_size=128, num_layers=2, input_seq_len=10, output_seq_len=5):
        """
        n_points: number of spatial positions per frame.
        input_seq_len: number of frames provided as input.
        output_seq_len: number of frames to predict.
        """
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.encoder = nn.LSTM(input_size=n_points, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=n_points, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_points)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        # input_seq shape: (batch, input_seq_len, n_points)
        batch_size = input_seq.size(0)
        _, (hidden, cell) = self.encoder(input_seq)
        # Initialize decoder input with the last frame of input_seq
        decoder_input = input_seq[:, -1, :].unsqueeze(1)  # shape: (batch, 1, n_points)
        outputs = []
        for t in range(self.output_seq_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out_frame = self.fc(out.squeeze(1))  # shape: (batch, n_points)
            outputs.append(out_frame.unsqueeze(1))
            # Use teacher forcing with the given probability
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t, :].unsqueeze(1)
            else:
                decoder_input = out_frame.unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, output_seq_len, n_points)
        return outputs


# ---------------------------
# Training Function
# ---------------------------
def train_model(model, dataloader, num_epochs=300, lr=0.0001, teacher_forcing_ratio=0.5):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, target_seq=targets, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    return model


# ---------------------------
# Testing Function
# ---------------------------
def test_model(model, input_seq):
    """
    Given an input tensor of shape (N_windows, input_seq_len, n_points),
    returns predictions of shape (N_windows, output_seq_len, n_points).
    """
    model.eval()
    if not isinstance(input_seq, torch.Tensor):
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
    input_seq = input_seq.to(device)
    with torch.no_grad():
        predictions = model(input_seq, target_seq=None, teacher_forcing_ratio=0.0)
        predictions = predictions.cpu().detach().numpy()
    return predictions


# ---------------------------
# Main Experiment
# ---------------------------
def run_experiment():
    # ----- Parameters -----
    input_seq_len = 10
    output_seq_len = 5
    stride_synthetic = 50  # For training we use a stride to limit the number of examples.
    N_points = 101  # Spatial resolution for synthetic data.
    N_samples = 100  # Number of synthetic samples.
    dt = 0.005
    t_start, t_end = 0, 10
    noise_std = 0.1

    # ----- 1. Synthetic Training Data Generation -----
    synthetic_inputs_list = []
    synthetic_targets_list = []
    for _ in range(N_samples):
        _, y_positions = generate_stick_swing_data(t_start, t_end, dt, N_points)
        # Normalize positions (each sample normalized independently)
        y_positions_noisy = y_positions + np.random.normal(loc=0.0, scale=noise_std, size=y_positions.shape)
        # Normalize positions (each sample normalized independently)
        y_norm = normalize(y_positions_noisy)
        X_win, Y_win = create_sliding_windows(y_norm, input_seq_len, output_seq_len, stride=stride_synthetic)
        synthetic_inputs_list.append(X_win)
        synthetic_targets_list.append(Y_win)
    # Concatenate all sliding window examples from all samples
    synthetic_inputs = np.concatenate(synthetic_inputs_list, axis=0)
    synthetic_targets = np.concatenate(synthetic_targets_list, axis=0)
    print("Synthetic training dataset shape:", synthetic_inputs.shape, synthetic_targets.shape)

    # Create DataLoader for training
    syn_dataset = TensorDataset(torch.tensor(synthetic_inputs, dtype=torch.float32),
                                torch.tensor(synthetic_targets, dtype=torch.float32))
    syn_loader = DataLoader(syn_dataset, batch_size=16, shuffle=True)

    # ----- 2. Train the Model on Synthetic Data -----
    model_0 = Seq2SeqLSTM(n_points=N_points, hidden_size=500, num_layers=2,
                          input_seq_len=input_seq_len, output_seq_len=output_seq_len).to(device)
    print("Training on synthetic data...")
    model_0 = train_model(model_0, syn_loader, num_epochs=10, lr=0.0001, teacher_forcing_ratio=0.5)

    # ----- Test on the Training Set ------------------

    pred_list_D0 = []
    Original_loss_list = []
    for i in range(synthetic_inputs.shape[0]):
        input_win = synthetic_inputs[i:i + 1]  # shape: (1, input_seq_len, n_points_mat)
        pred = test_model(model_0, input_win)  # shape: (1, output_seq_len, n_points_mat)
        pred_list_D0.append(pred[0])
    # Concatenate predictions from each window
    print(synthetic_targets.shape)
    Original_loss = compute_mse_over_time(synthetic_targets,pred_list_D0)
    print(Original_loss.shape)
    Original_loss_list.append(Original_loss[0])

    predicted_sequence_D0 = np.concatenate(pred_list_D0, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)
    # Also concatenate ground truth blocks (from Y_mat_test)
    gt_list_D0 = [synthetic_targets[i] for i in range(synthetic_targets.shape[0])]
    ground_truth_sequence_D0 = np.concatenate(gt_list_D0, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)

    # Create a time axis for the test sequence (normalized)
    # T_test_D0 = np.linspace(0, 1, ground_truth_sequence_D0.shape[0])
    T_test_D0 = np.linspace(0, 1, Original_loss.shape[0])

    # ----- 4. Compute CP Bound and Test MSE Over Time -----
    # Here, alpha = 0.1 means we select the loss at the rank corresponding to the top 10% highest errors.
    cp_bound_over_time = compute_cp_bound(ground_truth_sequence_D0, predicted_sequence_D0, alpha=0.1)
    mse_over_time = Original_loss

    print("CP Bound over time:", cp_bound_over_time.shape)
    print("Test MSE over time:", mse_over_time.shape)

    # ------------ load new data ----------------------------------
    # Load .mat file (make sure 'GP_Raw_data.mat' is in the same folder as this script)
    # TODO Adjust path as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Aligned_center_points.mat')
    mat_contents = sio.loadmat(file_path)
    x = mat_contents['x'].flatten('F')
    y = mat_contents['y'].flatten('F')
    z = mat_contents['z'].flatten('F')
    # Reshape: assume z holds time stamps and y contains the string positions
    time_steps_mat = np.unique(z)
    print(time_steps_mat.shape)
    n_points_mat = len(x) // len(time_steps_mat)
    y_reshaped = y.reshape(len(time_steps_mat), n_points_mat)
    # Normalize the positions
    y_mat_norm = normalize(y_reshaped)
    # Split into training (first 80%) and testing (last 20%)
    total_frames = y_mat_norm.shape[0]
    split_index = int(0.8 * total_frames)
    y_train_mat = y_mat_norm[:split_index, :]
    y_test_mat = y_mat_norm[split_index:, :]

    x_obv_test, y_obv_test = create_sliding_windows(y_mat_norm, input_seq_len, output_seq_len, stride=1)
    print("OBV test windows shape:", x_obv_test.shape, y_obv_test.shape)
    obv_pred_list = []
    for i in range(x_obv_test.shape[0]):
        input_win_obv = x_obv_test[i:i + 1]  # shape: (1, input_seq_len, n_points_mat)
        pred = test_model(model_0, input_win_obv)  # shape: (1, output_seq_len, n_points_mat)
        obv_pred_list.append(pred[0])
    # Concatenate predictions from each window
    obv_predicted_sequence = np.concatenate(obv_pred_list, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)
    New_loss = compute_mse_over_time(y_obv_test, obv_pred_list)

    # Also concatenate ground truth blocks (from Y_mat_test)
    obv_gt_list = [y_obv_test[i] for i in range(y_obv_test.shape[0])]
    obv_ground_truth_sequence = np.concatenate(obv_gt_list, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)
    obv_T_test = np.linspace(0, 1, obv_ground_truth_sequence.shape[0])
    T_test_D1 = np.linspace(0, 1, New_loss.shape[0])*20
    # ----- 5. Visualization -----
    plt.figure(figsize=(12, 8))
    time_steps = T_test_D0*20
    cp_bound_value = cp_bound_over_time  # assuming it's a single number
    plt.axhline(y=cp_bound_value, color='r', linestyle='--', label='Conformal Score C = {:.4f} (Threshold)'.format(cp_bound_value))
    plt.plot(time_steps, mse_over_time, marker='o', label='Test Data from the Original Dataset $D_0$')
    plt.plot(T_test_D1, New_loss, marker='x', label='Test Data from the New Dataset $D_1$')
    plt.xlabel('Time ($\mathrm{s}$)', fontsize=22, fontweight='bold')
    plt.ylabel('Conformal Score', fontsize=22, fontweight='bold')
    plt.title('CP Bound and Test Score from Original/New Dataset', fontsize=22)
    plt.legend( fontsize=19)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("CP_Bound_and_Test_Score.png", dpi=500, bbox_inches="tight")
    plt.show()


    # Create a time axis for the test sequence (normalized)


    # ----- 4. 3D Landscape Visualization -----
    # Spatial axis for .mat data (normalized)
    X_axis = np.linspace(0, 1, n_points_mat)
    X_grid, T_grid = np.meshgrid(X_axis, obv_T_test)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot ground truth surface
    surf_gt = ax.plot_surface(X_grid, T_grid, obv_ground_truth_sequence, cmap='viridis',
                              alpha=0.7, edgecolor='none')
    # Plot prediction surface
    surf_pred = ax.plot_surface(X_grid, T_grid, obv_predicted_sequence, cmap='autumn',
                                alpha=0.4, edgecolor='none')
    ax.set_xlabel('Spatial Position (normalized)', labelpad=10)
    ax.set_ylabel('Time (normalized)', labelpad=10)
    ax.set_zlabel('String Position (normalized)', labelpad=10)
    ax.set_title('3D Landscape: Ground Truth vs. LSTM Prediction (.mat Data)')

    legend_elements = [
        Patch(facecolor=surf_gt.get_facecolor()[0], edgecolor='none', label='New Observations'),
        Patch(facecolor=surf_pred.get_facecolor()[0], edgecolor='none', label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.view_init(elev=30, azim=45)
    plt.savefig("OOD_Landscape.png", dpi=500, bbox_inches="tight")
    plt.show()
    # # ------- This is the demo of OOD vs ID in 3D landscape ---------------------

    # ---- 3. Retraining on .mat Data -----
    print("y_train_mat shape:", y_train_mat.shape)
    X_mat_train, Y_mat_train = create_sliding_windows(y_train_mat, input_seq_len, output_seq_len, stride=1)
    print("MAT training windows shape:", X_mat_train.shape, Y_mat_train.shape)

    # Create DataLoader for training data
    mat_train_dataset = TensorDataset(torch.tensor(X_mat_train, dtype=torch.float32),
                                      torch.tensor(Y_mat_train, dtype=torch.float32))
    mat_train_loader = DataLoader(mat_train_dataset, batch_size=16, shuffle=True)

    # Initialize (or fine-tune) the model for .mat data
    model_mat = Seq2SeqLSTM(n_points=n_points_mat, hidden_size=500, num_layers=18,
                            input_seq_len=input_seq_len, output_seq_len=output_seq_len).to(device)
    print("Retraining model on .mat training data (first 80% of frames)...")
    model_mat = train_model(model_mat, mat_train_loader, num_epochs=300, lr=0.0001, teacher_forcing_ratio=0.5)

    # Build non-overlapping sliding windows for .mat test data (to reconstruct a full test sequence)
    X_mat_test, Y_mat_test = create_sliding_windows(y_test_mat, input_seq_len, output_seq_len, stride=output_seq_len)
    print("MAT test windows shape:", X_mat_test.shape, Y_mat_test.shape)

    # For each test window, predict the output block
    pred_list = []
    for i in range(X_mat_test.shape[0]):
        input_win = X_mat_test[i:i + 1]  # shape: (1, input_seq_len, n_points_mat)
        pred = test_model(model_mat, input_win)  # shape: (1, output_seq_len, n_points_mat)
        pred_list.append(pred[0])

    # Concatenate predictions from each window
    predicted_sequence = np.concatenate(pred_list, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)
    # Concatenate ground truth blocks (from Y_mat_test)
    gt_list = [Y_mat_test[i] for i in range(Y_mat_test.shape[0])]
    ground_truth_sequence = np.concatenate(gt_list, axis=0)  # shape: (num_windows * output_seq_len, n_points_mat)

    # Calculate Mean Squared Error (MSE) over time (averaging across spatial points)
    mse_per_time = np.mean((predicted_sequence - ground_truth_sequence) ** 2, axis=1)
    print('Shape for mse_per_time:', mse_per_time.shape)

    # Save the MSE over time as an Excel (.xls) file.
    # Make sure to install xlwt: pip install xlwt
    df = pd.DataFrame(mse_per_time)
    df.to_excel("LSTM_MSE.xls", index=False, header=False, engine='openpyxl')

    # Create a time axis for the test sequence (normalized)
    T_test = np.linspace(0, 1, ground_truth_sequence.shape[0])

    # ----- 4. 3D Landscape Visualization -----
    # Create spatial axis (normalized)
    X_axis = np.linspace(0, 1, n_points_mat)
    X_grid, T_grid = np.meshgrid(X_axis, T_test)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot ground truth surface
    surf_gt = ax.plot_surface(X_grid, T_grid, ground_truth_sequence, cmap='viridis',
                              alpha=0.7, edgecolor='none')
    # Plot prediction surface
    surf_pred = ax.plot_surface(X_grid, T_grid, predicted_sequence, cmap='autumn',
                                alpha=0.7, edgecolor='none')
    ax.set_xlabel('Spatial Position (normalized)', labelpad=10)
    ax.set_ylabel('Time (normalized)', labelpad=10)
    ax.set_zlabel('String Position (normalized)', labelpad=10)
    ax.set_title('3D Landscape: Ground Truth vs. LSTM Prediction (.mat Data)')
    legend_elements = [
        Patch(facecolor=surf_gt.get_facecolor()[0], edgecolor='none', label='Ground Truth'),
        Patch(facecolor=surf_pred.get_facecolor()[0], edgecolor='none', label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()

    # ----------------Visualize the Error ---------------------
    df = pd.read_excel("LSTM_MSE.xls", header=None, engine='openpyxl')

    # Convert the DataFrame to a 1D numpy array.
    mse_over_time = df.values.squeeze()

    # Create a time axis based on the length of the mse data.
    time_axis = range(len(mse_over_time))

    # Plot the MSE over time.
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mse_over_time, marker='o', linestyle='-')
    plt.xlabel("Time Step")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE Over Time")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_experiment()
