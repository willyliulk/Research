import jax
import jax.numpy as jnp
import flax
from jaxTraining.env.jax_env_3d import EnvParams, EnvState, reset_env, step_env
from jaxTraining.train_jax_ppo import ActorCritic
import numpy as np

def load_params(path):
    with open(path, "rb") as f:
        return flax.serialization.from_bytes(None, f.read())

def evaluate():
    try:
        params = load_params("jaxTraining/checkpoints_v4/model.msgpack")
    except:
        params = load_params("jaxTraining/checkpoints_v2/model.msgpack")
        
    network = ActorCritic(action_dim=2)
    env_params = EnvParams()
    
    key = jax.random.PRNGKey(42)
    num_envs = 1000
    keys = jax.random.split(key, num_envs)
    
    obsv, env_state = jax.vmap(reset_env, in_axes=(0, None))(keys, env_params)
    
    dones = jnp.zeros(num_envs, dtype=jnp.bool_)
    results = {"hit": 0, "crash": 0, "timeout": 0, "time_err": [], "reward": []}
    
    vmap_step = jax.vmap(step_env, in_axes=(0, 0, 0, None))
    
    for step in range(1000):
        key, step_key = jax.random.split(key)
        action_mean, action_logstd, value = network.apply(params, obsv)
        
        step_keys = jax.random.split(step_key, num_envs)
        next_obsv, next_env_state, reward, done, info = vmap_step(step_keys, env_state, action_mean, env_params)
        
        # Accumulate metrics for newly done envs
        just_done = done & ~dones
        
        if jnp.any(just_done):
            hits = info["hit"][just_done]
            crashes = info["crash"][just_done]
            timeouts = info["timeout"][just_done]
            time_errs = info["time_err"][just_done]
            rewards = info["reward"][just_done]
            
            results["hit"] += int(jnp.sum(hits))
            results["crash"] += int(jnp.sum(crashes))
            results["timeout"] += int(jnp.sum(timeouts))
            results["time_err"].extend(np.array(time_errs))
            results["reward"].extend(np.array(rewards))
            
        dones = dones | done
        obsv = next_obsv
        env_state = next_env_state
        
        if jnp.all(dones):
            break
            
    print(f"Total evaluated: {num_envs}")
    print(f"Hits: {results['hit']} ({(results['hit']/num_envs)*100:.1f}%)")
    print(f"Crashes: {results['crash']} ({(results['crash']/num_envs)*100:.1f}%)")
    print(f"Timeouts: {results['timeout']} ({(results['timeout']/num_envs)*100:.1f}%)")
    print(f"Avg Time Error: {np.mean(results['time_err']):.3f} s")
    print(f"Avg Final Reward: {np.mean(results['reward']):.1f}")

if __name__ == "__main__":
    evaluate()
