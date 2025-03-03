import jax
from jax import numpy as jnp
import brax
from brax import envs
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html
print(f"Using brax at: {brax.__file__}")
from tqdm import tqdm
from elbow import Elbow
from etils import epath

# Create PPO network and load parameters
ppo_network = ppo_networks.make_ppo_networks(
      4,
      6,
      preprocess_observations_fn=normalize)
model_path = 'elbow_params.pickle'
params = model.load_params(model_path)

def deterministic_policy(input_data: jnp.ndarray) -> jnp.ndarray:
    logits = ppo_network.policy_network.apply(*params[:2], jnp.array([input_data], dtype=jnp.float32))
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result[0] # remove the batch dimension

def get_obs(data, target: jnp.ndarray) -> jnp.ndarray:
    """Observes elbow angle, velocities, and last applied torque."""
    position = data.qpos

    # external_contact_forces are excluded
    return jnp.concatenate([
        position,
        data.qvel,
        data.qfrc_actuator,
        target
    ])

def main(is_msk: bool = True) -> None:
    envs.register_environment('elbow', Elbow)
    env = envs.get_environment('elbow', is_msk=is_msk)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(5)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    n_steps = 500
    for step in tqdm(range(n_steps), desc='Simulation'):
        if state.done or step % 100 == 0:
            state = jit_reset(state.info['rng'])
        obs = get_obs(state.pipeline_state, state.info['target_angle'])
        action = deterministic_policy(obs)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)
    
    # Save the trajectory as an HTML file
    html_content = html.render(
        env.sys.tree_replace({'opt.timestep': env.dt}),
        rollout,
        height='100vh',
        colab=False,
    )
    path = epath.Path('./elbow.html')
    path.write_text(html_content)

if __name__ == '__main__':
    main()