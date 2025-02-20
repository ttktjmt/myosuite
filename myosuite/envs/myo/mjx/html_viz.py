from datetime import datetime
import functools
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mujoco
from mujoco import mjx
from brax import envs
from brax.envs.base import Env, PipelineEnv, State
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import mjcf, model, html
import mujoco.viewer

from elbow import Elbow

print(f"Current backend: {jax.default_backend()}")

xml = '../assets/elbow/myoelbow_1dof6muscles_mjx_eval.xml'

ppo_network = ppo_networks.make_ppo_networks(
      4,
      6,
      preprocess_observations_fn=normalize)
model_path = 'elbow_params.pickle'
params = model.load_params(model_path)

def deterministic_policy(input_data):
    logits = ppo_network.policy_network.apply(*params[:2], jp.array([input_data], dtype=jp.float32))
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result[0] # remove the batch dimension

def get_obs(data, target):
    """Observes elbow angle, velocities, and last applied torque."""
    position = data.qpos

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        data.qvel,
        data.qfrc_actuator,
        target
    ])

def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    observations = get_obs(data, [data.ctrl[-1]])
    data.ctrl[:-1] = deterministic_policy(observations)
    pass

def main(is_msk=True):
    envs.register_environment('elbow', Elbow)
    env = envs.get_environment('elbow', is_msk=is_msk)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    n_steps = 500
    for step in range(n_steps):
        if state.done or step % 100 == 0:
            state = jit_reset(state.info['rng'])
        observations = get_obs(state.pipeline_state, state.info['target_angle'])
        action = deterministic_policy(observations)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)
    
    # Save the trajectory as an HTML file
    render_every = 2
    html_content = html.render(
        env.sys.tree_replace({'opt.timestep': env.dt}),
        rollout[::render_every],
        height=850,
    )
    with open('elbow.html', 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    main()