import jax
import jax.numpy as jnp
import numpy as np

from brax import jumpy as jp
from brax.io import mjcf, model  # model.load_params can still be used
from brax.io.html import render
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks

# Path to your MJCF file (Mujoco XML)
xml = '../assets/elbow/myoelbow_1dof6muscles_mjx_eval.xml'

# Create the PPO networks as before.
ppo_network = ppo_networks.make_ppo_networks(
    4, 6, preprocess_observations_fn=normalize)
model_path = 'elbow_params.pickle'
params = model.load_params(model_path)

def deterministic_policy(obs_np):
    """Compute the deterministic action given an observation.
    (Here we assume the network works with NumPy arrays.)
    """
    # Add batch dimension if needed.
    logits = ppo_network.policy_network.apply(
        *params[:2], np.array([obs_np], dtype=np.float32))
    action = ppo_network.parametric_action_distribution.mode(logits)
    # Remove batch dimension.
    return action[0]

# Load the MJCF file as a Brax system.
# This converts the Mujoco XML to a Brax config.
sys = mjcf.load(xml)

def get_obs(state):
    """Extract observation from a Brax state.
    Adjust this function so that the observation matches what your policy expects.
    For example, here we simply concatenate positions and velocities.
    """
    # In Brax, state.qp contains positions and velocities.
    # (Your system may require a different observation function.)
    return jnp.concatenate([state.qp[:state.sys.config.q_size],
                            state.qp[state.sys.config.q_size:]], axis=-1)

def rollout(sys, policy, episode_length=200):
    """Roll out one episode using the given policy on the Brax system."""
    key = jax.random.PRNGKey(0)
    state = sys.reset(key)
    trajectory = [state]
    for _ in range(episode_length):
        obs = get_obs(state)
        # Convert JAX array to NumPy before calling the policy.
        obs_np = np.array(obs)
        action = deterministic_policy(obs_np)
        # Convert the action back to a JAX array.
        state = sys.step(state, jnp.array(action))
        trajectory.append(state)
    return trajectory

# Run the rollout.
trajectory = rollout(sys, deterministic_policy, episode_length=200)

# Render the rollout to an interactive HTML string.
# (You can adjust height/width as needed.)
html_content = render(sys, trajectory, height=500, width=500)

# Save the HTML content to a file.
with open("rollout.html", "w") as f:
    f.write(html_content)

print("Rollout visualization saved as rollout.html")
