import flax.linen as nn
import jax
import jax.numpy as jnp

class ActorCritic(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Actor Network
        actor_x = nn.Dense(256)(x)
        actor_x = nn.swish(actor_x)
        actor_x = nn.Dense(256)(actor_x)
        actor_x = nn.swish(actor_x)
        actor_x = nn.Dense(256)(actor_x)
        actor_x = nn.swish(actor_x)
        action_mean = nn.Dense(self.action_dim)(actor_x)
        
        # Learnable standard deviation, independent of input state
        action_logstd = self.param('action_logstd', nn.initializers.zeros, (self.action_dim,))
        
        # Critic Network
        critic_x = nn.Dense(256)(x)
        critic_x = nn.swish(critic_x)
        critic_x = nn.Dense(256)(critic_x)
        critic_x = nn.swish(critic_x)
        critic_x = nn.Dense(256)(critic_x)
        critic_x = nn.swish(critic_x)
        value = nn.Dense(1)(critic_x)
        
        return action_mean, action_logstd, jnp.squeeze(value, axis=-1)

def sample_action(key, mean, logstd):
    """Samples action from Normal(mean, exp(logstd))"""
    std = jnp.exp(logstd)
    return mean + std * jax.random.normal(key, shape=mean.shape)

def get_logprob(action, mean, logstd):
    """Computes log probability of an action given the distribution parameters."""
    std = jnp.exp(logstd)
    var = std ** 2
    
    # Gaussian log-pdf
    log_prob = -((action - mean) ** 2) / (2 * var) - logstd - 0.5 * jnp.log(2 * jnp.pi)
    
    # Sum over action dimensions
    return jnp.sum(log_prob, axis=-1)

def get_entropy(logstd):
    """Computes entropy of the Gaussian distribution."""
    # Entropy of Gaussian is 0.5 * log(2 * pi * e * sigma^2)
    return jnp.sum(logstd + 0.5 + 0.5 * jnp.log(2 * jnp.pi), axis=-1)
