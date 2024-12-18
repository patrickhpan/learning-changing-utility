import numpy as np
import json
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import time

# np.random.seed(123)  # For reproducibility

# Read parameters from command line arguments with defaults
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=1,
                   help='Scale parameter for Cauchy distribution (utility changes)')
parser.add_argument('--sigma_epsilon', type=float, default=1, 
                   help='Scale parameter for the logistic error function')
parser.add_argument('--steps', type=int, default=500,
                   help='Number of observations')
parser.add_argument('--distortion_factor', type=float, default=1,
                   help='Distortion factor')
parser.add_argument('--playback', action='store_true', default=False,
                   help='Enable playback visualization')

args = parser.parse_args()

gamma = args.gamma
sigma_epsilon = args.sigma_epsilon
N_steps = args.steps
distortion_factor = args.distortion_factor

playback = args.playback
fps = 999.  # Frames per second (user-settable)


# Discretization of U:
# We'll choose a grid for U. Since utilities could in principle be unbounded,
# we choose a range that likely covers most of the probability mass.
U_min, U_max = -100, 100
U_density = 1 # points per unit interval
U_points = int((U_max - U_min) * U_density) + 1
U_grid = np.linspace(U_min, U_max, U_points)
dU = U_grid[1] - U_grid[0]


# -----------------------------------------
# Prior at t=0
# -----------------------------------------
# Uniform prior over grid size and U_min to U_max
# prior_pdf = np.ones(U_points) / (U_max - U_min)
prior_mu, prior_sigma = 0, (U_max - U_min) / 4  # 95% of mass within U_min to U_max
prior_pdf = norm.pdf(U_grid, prior_mu, prior_sigma)
prior_pdf /= np.trapezoid(prior_pdf, U_grid)  # Normalize just to be sure

current_pdf = prior_pdf.copy()
current_pdf_rational = prior_pdf.copy()  # Initialize rational estimator
posteriors_rational = []  # Store rational estimator posteriors

# Initialize l1 error histories
l1_errors_adaptive = []
l1_errors_rational = []
cumulative_error_adaptive = []
cumulative_error_rational = []
total_error_adaptive = 0.0
total_error_rational = 0.0

# -----------------------------------------
# Functions
# -----------------------------------------

def pi(u, w_val, sigma_eps):
    # Logistic choice probability of choosing u over w_val
    # pi(u,w) = 1/(1+exp((w-u)/sigma_epsilon))
    return 1.0 / (1.0 + np.exp((w_val - u)/sigma_eps))

def cauchy_pdf(delta, gamma_val):
    return (1/np.pi) * (gamma_val / (gamma_val**2 + delta**2))

def time_update(prev_pdf, U_grid, gamma_val):
    # Convolve prev_pdf with cauchy step PDF to get predictive distribution for U(t)
    # f_{U(t)}(u) = ∫ f_Δ(u-v) f_{U(t-1)}(v) dv
    # Using direct convolution (O(N^2)) approach here. For large N, could be optimized.
    N = len(U_grid)
    new_pdf = np.zeros(N)
    for i, u in enumerate(U_grid):
        # Evaluate integral: ∫ cauchy_pdf(u-v)*prev_pdf(v) dv
        # We'll use numerical integration with trapezoidal rule.
        # Let v = U_grid[j], delta = u - v
        delta_vals = u - U_grid
        cauchy_vals = cauchy_pdf(delta_vals, gamma_val)
        integrand = cauchy_vals * prev_pdf
        new_pdf[i] = np.trapezoid(integrand, U_grid)
    # Normalize
    new_pdf /= np.trapezoid(new_pdf, U_grid)
    return new_pdf

def update_with_observation(pred_pdf, U_grid, observation, w_val, sigma_eps):
    # observation: either "succ" or "prec"
    # If "succ": we observed that X was chosen over w_val
    # likelihood(u) = pi(u, w_val) if succ, else pi(w_val, u)
    # Note pi(w,u) = 1 - pi(u,w)

    if observation == "succ":
        likelihood = pi(U_grid, w_val, sigma_eps)
    else:
        likelihood = 1.0 - pi(U_grid, w_val, sigma_eps)

    post_unnormalized = pred_pdf * likelihood
    normalization = np.trapezoid(post_unnormalized, U_grid)
    if normalization == 0:
        # Avoid division by zero; reset to uniform if necessary
        post_pdf = prior_pdf.copy()
    else:
        post_pdf = post_unnormalized / normalization
    return post_pdf

def update_rational_estimator(current_pdf_rational, U_grid, observation, w_val):
    # Update the rational estimator based on the observation
    if observation == "succ":
        # X chosen over w_val implies U(t) > w_val
        mask = U_grid > w_val
    else:
        # "prec" implies w_val chosen over X, so U(t) < w_val
        mask = U_grid < w_val

    new_pdf = current_pdf_rational * mask
    normalization = np.trapezoid(new_pdf, U_grid)
    if normalization == 0:
        # If no support after truncation, revert to the original distribution
        # new_pdf = prior_pdf.copy()
        # Set the new pdf to a copy of the prior_pdf but with the current median
        # new_pdf = prior_pdf.copy()
        current_median = np.median(U_grid[current_pdf_rational > 0])
        # median_idx = np.searchsorted(U_grid, current_median)
        # new_pdf = np.roll(new_pdf, median_idx - np.searchsorted(U_grid, 0))
        new_pdf = norm.pdf(U_grid, current_median, prior_sigma)
        # print(f"Rational estimator could not be updated at step with obs '{observation}' and w_val {w_val}. Reverting to original prior distribution.")
    else:
        # Normalize the truncated distribution
        new_pdf /= normalization
    return new_pdf

# -----------------------------------------
# Agent Class
# -----------------------------------------
class Agent:
    def __init__(self, U_min, U_max, true_gamma, true_sigma_eps, distortion_factor=0):
        # Apply distortion to true gamma and sigma_eps
        self.gamma = true_gamma * 10**np.random.uniform(-distortion_factor, distortion_factor)
        self.sigma_eps = true_sigma_eps * 10**np.random.uniform(-distortion_factor, distortion_factor)
        self.U_min = U_min
        self.U_max = U_max
        self.true_utility = 0.0  # Initial true utility

    def evolve_utility(self):
        # Evolve the true utility by drawing from Cauchy(0, gamma)
        delta = cauchy.rvs(loc=0, scale=self.gamma)
        self.true_utility += delta
        # Ensure the utility stays within the defined bounds
        self.true_utility = min(max(self.U_min, self.true_utility), self.U_max)
        return self.true_utility

    def bid_result(self, bid):
        # Determine the true observation based on the current true utility and the bid
        succ_prob = 1 / (1 + np.exp((bid - self.true_utility) / self.sigma_eps))
        observed_obs = 'succ' if np.random.rand() < succ_prob else 'prec'

        return observed_obs

# Create an agent with distortion
agent = Agent(U_min, U_max, gamma, sigma_epsilon, distortion_factor)

# -----------------------------------------
# Main Iteration and Data Generation
# -----------------------------------------




posteriors = []
observations = []
true_u_history = []
err_history = []
posteriors_rational = []
median_history = []
median_history_rational = []

current_pdf = prior_pdf.copy()
current_pdf_rational = prior_pdf.copy()  # Initialize rational estimator

print("Generating data...")

def get_bid(current_pdf):
    # Compute the median of the current posterior distribution
    cumulative_pdf = np.cumsum(current_pdf) * dU
    median_index = np.searchsorted(cumulative_pdf, 0.5)
    median_value = U_grid[median_index]
    return median_value

for t in range(1, N_steps + 1):
    # Agent makes a bid based on the median of the current posterior
    adaptive_bid = get_bid(current_pdf)
    rational_bid = get_bid(current_pdf_rational)

    # Agent generates an observation based on the bid and its true utility
    adaptive_obs = agent.bid_result(adaptive_bid)
    rational_obs = agent.bid_result(rational_bid)

    observations.append((adaptive_obs, adaptive_bid))
    true_u_history.append(agent.true_utility)
    err_history.append(adaptive_obs != ('succ' if agent.true_utility > adaptive_bid else 'prec'))

    # Evolve the agent's true utility for the next step
    agent.evolve_utility()

    # Prediction step: evolve U(t-1) into U(t)
    pred_pdf = time_update(current_pdf, U_grid, gamma)

    # Update step: incorporate observation
    current_pdf = update_with_observation(pred_pdf, U_grid, adaptive_obs, adaptive_bid, sigma_epsilon)
    posteriors.append((t, adaptive_obs, adaptive_bid, current_pdf.copy()))

    # Update rational estimator using the new method
    current_pdf_rational = update_rational_estimator(current_pdf_rational, U_grid, rational_obs, rational_bid)
    posteriors_rational.append(current_pdf_rational.copy())

    # Compute median predictions
    cumulative_pdf = np.cumsum(current_pdf) * dU
    median_idx = np.searchsorted(cumulative_pdf, 0.5)
    median_val = U_grid[median_idx]
    median_history.append(median_val)

    cumulative_pdf_rational = np.cumsum(current_pdf_rational) * dU
    median_idx_rational = np.searchsorted(cumulative_pdf_rational, 0.5)
    median_val_rational = U_grid[median_idx_rational]
    median_history_rational.append(median_val_rational)

    # Compute l1 errors
    true_u = true_u_history[-1]
    error_adaptive = np.abs(true_u - median_val)
    error_rational = np.abs(true_u - median_val_rational)
    # error_adaptive = (true_u - median_val)**2
    # error_rational = (true_u - median_val_rational)**2
    l1_errors_adaptive.append(error_adaptive)
    l1_errors_rational.append(error_rational)

    # Update cumulative errors
    total_error_adaptive += error_adaptive
    total_error_rational += error_rational
    cumulative_error_adaptive.append(total_error_adaptive)
    cumulative_error_rational.append(total_error_rational)

    # Print summary statistics for debugging every 100 steps
    if t % 100 == 0:
        mean_u = np.trapezoid(U_grid * current_pdf, U_grid)
        var_u = np.trapezoid((U_grid - mean_u)**2 * current_pdf, U_grid)
        mean_u_rational = np.trapezoid(U_grid * current_pdf_rational, U_grid)
        var_u_rational = np.trapezoid((U_grid - mean_u_rational)**2 * current_pdf_rational, U_grid)
        print(f"Generated {t} of {N_steps} steps:")
        # print(f"  Adaptive Estimator - Mean U(t): {mean_u:.4f}, Std: {np.sqrt(var_u):.4f}")
        # print(f"  Rational Estimator - Mean U(t): {mean_u_rational:.4f}, Std: {np.sqrt(var_u_rational):.4f}")

print("Data generation complete.")

# -----------------------------------------
# Playback of Generated Data
# -----------------------------------------

# Determine maximum y value across all posteriors for consistent y-axis scaling
max_pdf = min(1.0, max([pdf.max() for (_, _, _, pdf) in posteriors] + [pdf.max() for pdf in posteriors_rational]))
max_pdf *= 1.05  # Add 5% buffer

# Setup Plot with three subplots: posterior, true utility over time, and error metrics
fig, (ax_pdf, ax_true, ax_error) = plt.subplots(3, 1, figsize=(7, 11))
plt.subplots_adjust(hspace=0.4, left=0.1)

# Initialize posterior plot
initial_t, initial_obs, initial_w_val, initial_pdf = posteriors[0]
initial_pdf_rational = posteriors_rational[0]
line_pdf, = ax_pdf.plot(U_grid, initial_pdf, color='blue', label='Adaptive Posterior')
line_pdf_rational, = ax_pdf.plot(U_grid, initial_pdf_rational, color='magenta', label='Rational Posterior')
w_line = ax_pdf.axvline(x=initial_w_val, color='red', linestyle='--', label='w_val')
true_utility_line = ax_pdf.axvline(x=true_u_history[0], color='green', linestyle='--', label='True Utility')
median_value = U_grid[np.searchsorted(np.cumsum(initial_pdf)*dU, 0.5)]
median_line = ax_pdf.axvline(x=median_value, color='purple', linestyle='--', label=f'Adaptive Median={median_val:.4f}')
median_value_rational = U_grid[np.searchsorted(np.cumsum(initial_pdf_rational)*dU, 0.5)]
median_line_rational = ax_pdf.axvline(x=median_value_rational, color='magenta', linestyle='--', label=f'Rational Median={median_val_rational:.4f}')

# Add labels for true sigma_epsilon and gamma as well as agent's values
ax_pdf.text(0.02, 0.95, f'True sigma_epsilon: {sigma_epsilon}', transform=ax_pdf.transAxes, fontsize=10, verticalalignment='top')
ax_pdf.text(0.02, 0.90, f'True gamma: {gamma}', transform=ax_pdf.transAxes, fontsize=10, verticalalignment='top')
ax_pdf.text(0.02, 0.85, f'Agent sigma_epsilon: {agent.sigma_eps:.4f}', transform=ax_pdf.transAxes, fontsize=10, verticalalignment='top')
ax_pdf.text(0.02, 0.80, f'Agent gamma: {agent.gamma:.4f}', transform=ax_pdf.transAxes, fontsize=10, verticalalignment='top')

ax_pdf.set_title(f'Posterior Distributions at Time 1\nObservation: {initial_obs} vs w={initial_w_val}\nTrue Utility: {true_u_history[0]:.4f}\nError: {err_history[0]}')
ax_pdf.set_xlabel('Utility U(t)')
ax_pdf.set_ylabel('Probability Density')
ax_pdf.set_ylim(0, max_pdf)
ax_pdf.legend()
ax_pdf.grid(True)

# Initialize true utility over time plot
true_utility_plot, = ax_true.plot([], [], color='green', label='True Utility Over Time', alpha=0.7)
median_utility_plot, = ax_true.plot([], [], color='orange', label='Adaptive Median Utility', alpha=0.7)
median_utility_plot_rational, = ax_true.plot([], [], color='magenta', label='Rational Median Utility', alpha=0.7)
error_lines = [ax_true.axvline(x=t, color='red', alpha=0.2, linewidth=0.5) for t, err in enumerate(err_history) if err]
ax_true.set_title('True Utility Over Time')
ax_true.set_xlabel('Time Step')
ax_true.set_ylabel('Utility U(t)')
ax_true.set_xlim(1, N_steps)
ax_true.set_ylim(U_min, U_max)
ax_true.legend()
ax_true.grid(True)

# Initialize error metrics plot
line_error_adaptive, = ax_error.plot([], [], color='blue', label='Adaptive l1 Error')
line_error_rational, = ax_error.plot([], [], color='magenta', label='Rational l1 Error')
line_cumulative_adaptive, = ax_error.plot([], [], color='blue', linestyle='--', label='Adaptive Cumulative Error')
line_cumulative_rational, = ax_error.plot([], [], color='magenta', linestyle='--', label='Rational Cumulative Error')
ax_error.set_title('l1 Error Between True Utility and Predicted Utilities')
ax_error.set_xlabel('Time Step')
ax_error.set_ylabel('Cumulative l1 Error')
ax_error.set_xlim(1, N_steps)
ax_error.set_ylim(0, max(max(l1_errors_adaptive, default=1), max(l1_errors_rational, default=1)) * 1.1)
ax_error.legend()
ax_error.grid(True)

# Create a secondary y-axis for instantaneous l1 error
ax_error_instant = ax_error.twinx()
line_error_adaptive_instant, = ax_error_instant.plot([], [], color='blue', linestyle=':', label='Adaptive Instantaneous l1 Error')
line_error_rational_instant, = ax_error_instant.plot([], [], color='magenta', linestyle=':', label='Rational Instantaneous l1 Error')
ax_error_instant.set_ylabel('Instantaneous l1 Error')
ax_error_instant.set_ylim(0, max(max(l1_errors_adaptive, default=1), max(l1_errors_rational, default=1)) * 1.1)
ax_error_instant.legend(loc='upper right')

# Initialize median histories
median_history = []
median_history_rational = []

# Initialize error histories for plotting
error_history_adaptive = []
error_history_rational = []
cumulative_error_history_adaptive = []
cumulative_error_history_rational = []

plt.tight_layout()

# Function to update the plots for each frame
def update_plot(t_idx):
    t, obs, w_val, pdf = posteriors[t_idx]
    pdf_rational = posteriors_rational[t_idx]

    # Update adaptive posterior plot
    line_pdf.set_ydata(pdf)

    # Update rational posterior plot
    line_pdf_rational.set_ydata(pdf_rational)

    # Update w_val line
    w_line.set_xdata([w_val, w_val])
    w_line.set_label(f'w_val={w_val:.4f}')

    # Update true utility line in posterior plot
    true_u = true_u_history[t_idx]
    true_utility_line.set_xdata([true_u, true_u])
    true_utility_line.set_label(f'True Utility={true_u:.4f}')

    # Update median lines for adaptive estimator
    cumulative_pdf = np.cumsum(pdf) * dU
    median_idx = np.searchsorted(cumulative_pdf, 0.5)
    median_val = U_grid[median_idx]
    median_line.set_xdata([median_val, median_val])
    median_line.set_label(f'Adaptive Median={median_val:.4f}')

    # Update median lines for rational estimator
    cumulative_pdf_rational = np.cumsum(pdf_rational) * dU
    median_idx_rational = np.searchsorted(cumulative_pdf_rational, 0.5)
    median_val_rational = U_grid[median_idx_rational]
    median_line_rational.set_xdata([median_val_rational, median_val_rational])
    median_line_rational.set_label(f'Rational Median={median_val_rational:.4f}')

    # Append median values to history
    median_history.append(median_val)
    median_history_rational.append(median_val_rational)
    median_utility_plot.set_data(range(1, t_idx + 2), median_history)
    median_utility_plot_rational.set_data(range(1, t_idx + 2), median_history_rational)

    # Update posterior plot title
    ax_pdf.set_title(f'Posterior Distributions at Time {t + 1}\nObservation: {obs} vs w={w_val}\nTrue Utility: {true_u:.4f}\nError: {err_history[t_idx]}')

    # Update legends to reflect labels
    ax_pdf.legend()
    ax_true.legend()

    # Update true utility over time plot
    true_utility_plot.set_data(range(1, t_idx + 2), true_u_history[:t_idx + 1])

    # Update error metrics plot
    error_adaptive = l1_errors_adaptive[t_idx]
    error_rational = l1_errors_rational[t_idx]
    cumulative_adaptive = cumulative_error_adaptive[t_idx]
    cumulative_rational = cumulative_error_rational[t_idx]

    error_history_adaptive.append(error_adaptive)
    error_history_rational.append(error_rational)
    cumulative_error_history_adaptive.append(cumulative_adaptive)
    cumulative_error_history_rational.append(cumulative_rational)

    line_error_adaptive.set_data(range(1, t_idx + 2), cumulative_error_history_adaptive)
    line_error_rational.set_data(range(1, t_idx + 2), cumulative_error_history_rational)
    line_error_adaptive_instant.set_data(range(1, t_idx + 2), error_history_adaptive)
    line_error_rational_instant.set_data(range(1, t_idx + 2), error_history_rational)

    # Adjust y-axis if necessary
    current_max = max(cumulative_adaptive, cumulative_rational, 
                      max(line_error_adaptive.get_ydata(), default=0),
                      max(line_error_rational.get_ydata(), default=0),
                      max(line_error_adaptive_instant.get_ydata(), default=0),
                      max(line_error_rational_instant.get_ydata(), default=0))
    ax_error.set_ylim(0, current_max * 1.1 if current_max > 0 else 1)
    ax_error_instant.set_ylim(0, current_max * 1.1 if current_max > 0 else 1)

    # Update error plot title if needed
    ax_error.set_title('l1 Error Between True Utility and Predicted Utilities')

    # Update legends to reflect labels
    ax_error.legend()
    ax_error_instant.legend()

    # Redraw the canvas
    fig.canvas.draw_idle()
    

# Playback control variables
paused = False
current_frame = 0

# def on_key(event):
#     global paused, current_frame
#     if event.key == ' ':
#         paused = not paused
#     elif event.key == 'right':
#         current_frame = min(current_frame + 1, N_steps - 1)
#         update_plot(current_frame)
#     elif event.key == 'left':
#         current_frame = max(current_frame - 1, 0)
#         update_plot(current_frame)

# fig.canvas.mpl_connect('key_press_event', on_key)

# Playback the data at the specified frame rate
if playback is True:
    print("Starting playback...")
    while current_frame < N_steps:
        if not plt.fignum_exists(fig.number):
            print("Figure closed by user. Exiting playback.")
            break
        if not paused:
            update_plot(current_frame)
            current_frame += 1
            plt.pause(1.0 / fps)
        else:
            plt.pause(0.1)

    print("Playback complete.")
    plt.show()
# Print true and agent parameters
print("\nTrue vs Agent Parameters:")
print(f"  True sigma_epsilon: {sigma_epsilon}")
print(f"  Agent sigma_epsilon: {agent.sigma_eps:.4f}")
print(f"  True gamma: {gamma}")
print(f"  Agent gamma: {agent.gamma:.4f}")

# Print final cumulative L1 losses
print(f"\nFinal Cumulative L1 Loss:")
print(f"  Adaptive Estimator: {total_error_adaptive:.4f}")
print(f"  Rational Estimator: {total_error_rational:.4f}")


results = {
    "true": {
        "sigma_epsilon": sigma_epsilon,
        "gamma": gamma,
        "distortion_factor": distortion_factor,
    },
    "agent": {
        "sigma_epsilon": agent.sigma_eps,
        "gamma": agent.gamma,
    },
    "cumulative_loss": {
        "adaptive": total_error_adaptive,
        "rational": total_error_rational,
        "adaptive_advantage": total_error_rational / total_error_adaptive,
    },
    "steps": N_steps
}

print("\nResults as JSON:")
print(json.dumps(results))
