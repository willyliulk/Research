import os
import sys
import asyncio
from pathlib import Path
import time
import datetime
import argparse
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import numpy as np
import torch
import flax.linen as nn
import requests

from env.jax_env_target_3d import EnvParams, EnvState, reset_env, step_env, get_uav_obs

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))# from env.jax_env_3d import EnvState, EnvParams, step_env, reset_env
from models.ppo_network import ActorCritic, sample_action, get_logprob, get_entropy

# =====================================================================
# Hyperparams
# =====================================================================
NUM_ENVS = 2048
NUM_STEPS = 64
TOTAL_TIMESTEPS = 1000_000_000
NUM_EPOCHS = 5
MINIBATCH_SIZE = 4096
GAMMA = 0.9995
GAE_LAMBDA = 0.99
CLIP_EPS = 0.2
VF_COEF = 0.5
LR = 1e-4

# Leaky PPO Hyperparameter
LEAKY_ALPHA = 0.01

UPDATES_PER_CHUNK = 10  # 每次 JIT 執行 10 次更新，然後回傳 Python 層級以便回報進度與早停判定

BOT_TOKEN = "7326959783:AAH6CLBJZog_35ixc11d_SO6RibrhMTlg_A"
CHAT_ID = "5575073351"

import requests
def tg_send_msg(msg: str):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print("Telegram 通知失敗:", e)

class TrainState(train_state.TrainState):
    pass

def make_train_chunk(num_envs, num_steps):
    minibatch_size = MINIBATCH_SIZE
    if num_envs * num_steps < minibatch_size:
        minibatch_size = num_envs * num_steps
    num_minibatches = (num_envs * num_steps) // minibatch_size
    
    config = {
        "LR": LR,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "UPDATE_EPOCHS": NUM_EPOCHS,
        "NUM_MINIBATCHES": num_minibatches,
        "GAMMA": GAMMA,
        "GAE_LAMBDA": GAE_LAMBDA,
        "CLIP_EPS": CLIP_EPS,
        "LEAKY_ALPHA": LEAKY_ALPHA,
        "ENT_COEF": 0.001,
        "VF_COEF": VF_COEF,
        "MINIBATCH_SIZE": minibatch_size,
    }
    
    network = TargetActorCritic(action_dim=2)
    
    def train_chunk(runner_state, env_params, uav_params):
        vmap_step = jax.vmap(step_env, in_axes=(0, 0, 0, 0, None))
        vmap_uav_obs = jax.vmap(get_uav_obs, in_axes=(0, None))
        uav_network = ActorCritic(action_dim=2)
        
        def _update_step(runner_state, unused):
            state, env_state, obsv, rng = runner_state
            
            def _env_step(runner_state, unused):
                state, env_state, obsv, rng = runner_state
                
                rng, pi_rng = jax.random.split(rng)
                action_mean, action_logstd, value = network.apply(state.params, obsv)
                action = sample_action(pi_rng, action_mean, action_logstd)
                log_prob = get_logprob(action, action_mean, action_logstd)
                
                rng, step_rng = jax.random.split(rng)
                step_rng_split = jax.random.split(step_rng, config["NUM_ENVS"])
                
                uav_obsv = vmap_uav_obs(env_state, env_params)
                uav_action_mean, _, _ = uav_network.apply(uav_params, uav_obsv)
                next_obsv, next_env_state, reward, done, info = vmap_step(step_rng_split, env_state, action, uav_action_mean, env_params)
                
                # ==== Auto-Reset ====
                reset_obsv, reset_env_state = jax.vmap(reset_env, in_axes=(0, None))(step_rng_split, env_params)
                
                def _where_done(done_cond, reset_val, next_val):
                    # Broadcast done_cond to match reset_val shape
                    expand_dims = tuple(range(1, reset_val.ndim))
                    done_expanded = jnp.expand_dims(done_cond, axis=expand_dims) if expand_dims else done_cond
                    return jnp.where(done_expanded, reset_val, next_val)

                next_env_state = jax.tree_util.tree_map(
                    lambda x, y: _where_done(done, x, y),
                    reset_env_state,
                    next_env_state
                )
                
                next_obsv = jax.tree_util.tree_map(
                    lambda x, y: _where_done(done, x, y),
                    reset_obsv,
                    next_obsv
                )
                # ====================
                
                transition = {
                    "done": done,
                    "action": action,
                    "value": value,
                    "reward": reward,
                    "log_prob": log_prob,
                    "obs": obsv,
                    "info": info
                }
                runner_state = (state, next_env_state, next_obsv, rng)
                return runner_state, transition
                
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            
            state, env_state, obsv, rng = runner_state
            _, _, last_val = network.apply(state.params, obsv)
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition["done"], transition["value"], transition["reward"]
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch["value"]
                
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
                        action_mean, action_logstd, value = network.apply(params, traj_batch["obs"])
                        log_prob = get_logprob(traj_batch["action"], action_mean, action_logstd)
                        entropy = get_entropy(action_logstd)
                        
                        logratio = log_prob - traj_batch["log_prob"]
                        ratio = jnp.exp(logratio)
                        
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        
                        # ========= Leaky PPO Implementation =========
                        alpha = config["LEAKY_ALPHA"]
                        eps = config["CLIP_EPS"]
                        
                        # Upper and lower bounds dependent on ratio and alpha
                        l_sa = alpha * ratio + (1 - alpha) * (1 - eps)
                        u_sa = alpha * ratio + (1 - alpha) * (1 + eps)
                        
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, l_sa, u_sa) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        # ============================================
                        
                        loss_value = jnp.mean((value - targets) ** 2)
                        loss_entropy = -entropy.mean()
                        
                        total_loss = loss_actor + config["VF_COEF"] * loss_value + config["ENT_COEF"] * loss_entropy
                        return total_loss, (loss_actor, loss_value, loss_entropy)
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss_info, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, permutation_rng = jax.random.split(rng)
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(permutation_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
                
                train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info
                
            update_state = (state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            state = update_state[0]
            rng = update_state[-1]
            
            # metrics from the last step in trajectories
            mean_step_reward = traj_batch["reward"].mean()
            
            dones = traj_batch["done"]
            ep_lens = traj_batch["info"]["episode_len"]
            done_count = jnp.sum(dones)
            mean_ep_len = jnp.where(done_count > 0, jnp.sum(ep_lens * dones) / done_count, 0.0)
            
            metrics = {
                "mean_step_reward": mean_step_reward,
                "mean_ep_len": mean_ep_len,
            }
            runner_state = (state, env_state, obsv, rng)
            return runner_state, metrics
            
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, UPDATES_PER_CHUNK)
        return runner_state, metrics

    return train_chunk

def init_runner_state(rng, num_envs, env_params, load_model_path=None, total_updates=0):
    network = TargetActorCritic(action_dim=2)
    
    rng, init_rng = jax.random.split(rng)
    dummy_obs = {
        "current": jnp.zeros((1, 6)),
        "history": jnp.zeros((1, 25, 3))
    }
    network_params = network.init(init_rng, dummy_obs)
    
    if load_model_path is not None and os.path.exists(load_model_path):
        print(f"從 {load_model_path} 載入預訓練模型...")
        with open(load_model_path, "rb") as f:
            network_params = flax.serialization.from_bytes(network_params, f.read())
    
    if total_updates > 0:
        # PPO epoch optimization means apply_gradients is called multiple times per update
        num_minibatches = max(1, (NUM_ENVS * NUM_STEPS) // MINIBATCH_SIZE)
        total_opt_steps = total_updates * NUM_EPOCHS * num_minibatches
        schedule = optax.linear_schedule(init_value=LR, end_value=1e-6, transition_steps=total_opt_steps)
        tx = optax.adam(learning_rate=schedule)
    else:
        tx = optax.adam(learning_rate=LR)
        
    state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    
    rng, reset_rng = jax.random.split(rng)
    reset_rng_split = jax.random.split(reset_rng, num_envs)
    obsv, env_state = jax.vmap(reset_env, in_axes=(0, None))(reset_rng_split, env_params)
    
    return (state, env_state, obsv, rng)

from models.target_ppo_network import TargetActorCritic
def save_checkpoint(params, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    msgpack_path = os.path.join(save_dir, "model.msgpack")
    with open(msgpack_path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
        
    # Target ONNX 導出
    try:
        from models.target_ppo_network import export_to_onnx
        export_to_onnx(params, save_dir)
    except Exception as e:
        print(f"⚠️ ONNX 模型導出失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description="JAX PPO Training")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test")
    parser.add_argument("--load_model", type=str, default=None, help="Path to model.msgpack to resume training")
    parser.add_argument("--phase", type=str, default="auto", help="Training phase: 1, 2, or auto")
    parser.add_argument("--profile", action="store_true", help="Enable JAX profiler for the 3rd chunk")
    args = parser.parse_args()

    dry_run = args.dry_run
    load_model_path = args.load_model
    phase_mode = args.phase
    profile_mode = args.profile
    
    if load_model_path:
        save_dir = os.path.dirname(load_model_path)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        save_dir = os.path.join(os.path.dirname(__file__), "checkpoints", f"checkpoint_target_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    
    if dry_run:
        num_envs = 64
        num_steps = 32
        total_timesteps = num_envs * num_steps * 20
        global UPDATES_PER_CHUNK
        UPDATES_PER_CHUNK = 2
    else:
        num_envs = NUM_ENVS
        num_steps = NUM_STEPS
        total_timesteps = TOTAL_TIMESTEPS

    print("====== 初始化 Leaky PPO (純 JAX) 訓練 ======")
    print(f"硬體資訊: {jax.devices()}")
    print(f"訓練設定: {num_envs} 總環境, {num_steps} 步長/次, {UPDATES_PER_CHUNK} 更新/輪")
    
    total_updates = total_timesteps // (num_envs * num_steps)
    current_phase = 1 if phase_mode in ["1", "auto"] else 2
    
    initial_env_params = EnvParams()
    
    rng = jax.random.PRNGKey(42)
    runner_state = init_runner_state(rng, num_envs, initial_env_params, load_model_path=load_model_path, total_updates=total_updates)
    
    print("====== 載入 UAV 預訓練模型 ======")
    uav_model_path = "jaxTraining/checkpoints/checkpoint_20260706-1701/model.msgpack"
    with open(uav_model_path, "rb") as f:
        uav_params = flax.serialization.msgpack_restore(f.read())
        
    train_chunk_fn = make_train_chunk(num_envs, num_steps)
    train_chunk_jit = jax.jit(train_chunk_fn)
    
    # Trigger JIT compilation
    print("編譯與執行中 (JIT 編譯大約需要 1~2 分鐘)...")
    start_t = time.time()
    
    # Compile with initial params
    _ = train_chunk_jit(runner_state, initial_env_params, uav_params)
    
    total_chunks = total_updates // UPDATES_PER_CHUNK
    
    early_stop_target = 5.0 # Minimum reward to consider early stopping
    best_reward = -999.0
    recent_rewards = []
    
    tg_send_msg(f"🚀 3D Leaky PPO 訓練啟動\n外層迴圈: Python\n內層編譯: JAX JIT\nChunk 更新數: {UPDATES_PER_CHUNK}")
    
    try:
        for chunk_idx in range(total_chunks):
            env_params = EnvParams()
            
            if profile_mode and chunk_idx == 2:
                print("\n[Profiler] 開始記錄 GPU Trace...")
                jax.profiler.start_trace("/tmp/tensorboard_jax")
                
            runner_state, metrics = train_chunk_jit(runner_state, env_params, uav_params)
            jax.block_until_ready(runner_state)
            
            if profile_mode and chunk_idx == 2:
                jax.profiler.stop_trace()
                print(f"[Profiler] 記錄完成！請使用 `tensorboard --logdir=/tmp/tensorboard_jax` 查看")
                import sys; sys.exit(0) # 紀錄完直接退出，不繼續訓練
            
            # Extract metrics (average over the chunk)
            mean_step_reward = float(np.mean(metrics["mean_step_reward"]))
            mean_ep_len = float(np.mean(metrics["mean_ep_len"]))
            
            # Extract step from train_state (runner_state[0])
            current_opt_step = int(runner_state[0].step)
            num_minibatches = max(1, (num_envs * num_steps) // MINIBATCH_SIZE)
            total_opt_steps = total_updates * NUM_EPOCHS * num_minibatches
            
            # Compute current LR based on linear schedule
            current_lr = LR - (LR - 1e-7) * min(1.0, current_opt_step / total_opt_steps)
            
            current_update = (chunk_idx + 1) * UPDATES_PER_CHUNK
            elapsed = time.time() - start_t
            sps = (current_update * num_envs * num_steps) / elapsed
            
            print(f"[{current_update}/{total_updates} updates] | 步長報酬: {mean_step_reward:.4f} | 平均存活: {mean_ep_len:.1f} | SPS: {sps:.0f} | LR: {current_lr:.2e}")
            
            if mean_step_reward > best_reward:
                best_reward = mean_step_reward
                
            # Curriculum Learning 自動切換
            if phase_mode == "auto" and current_phase == 1:
                if mean_step_reward > 3.8 and current_update > 20:
                    msg = "🚀 自動課程學習觸發: 模型已學會基礎尋標，自動切換至 Phase 2 (時間控制)！"
                    print(f"\n{msg}")
                    tg_send_msg(msg)
                    current_phase = 2
                    recent_rewards.clear() # 清空早停歷史紀錄，重新適應新環境
                    best_reward = -999.0
                    continue
                    
            recent_rewards.append(mean_step_reward)
            if len(recent_rewards) > 200:
                recent_rewards.pop(0)
                
            # 早停機制: 連續 200 次的標準差過小(代表完全不再上升或變動)
            if len(recent_rewards) == 200 and current_update > 500:
                reward_std = float(np.std(recent_rewards))
                if reward_std < 0.02:
                    msg = f"🔔 觸發自動早停！模型已完全收斂，停止上升 (Std: {reward_std:.4f}, 報酬: {mean_step_reward:.4f})"
                    print(f"\n[Stop] {msg}")
                    tg_send_msg(msg)
                    break
                
    except KeyboardInterrupt:
        print("\n[Interrupt] 監聽到 Ctrl+C 中斷，安全存檔並退出中...")
        tg_send_msg("⚠️ 訓練已被使用者手動中斷，保存目前最新模型。")
        
    end_t = time.time()
    
    print(f"訓練結束！總耗時: {end_t - start_t:.2f} 秒")
    if not dry_run:
        tg_send_msg(f"✅ 3D Leaky PPO 訓練完成\n最高報酬: {best_reward:.4f}\n耗時: {end_t - start_t:.2f} s\n路徑: {save_dir}")
        save_checkpoint(runner_state[0].params, save_dir)
        print(f"模型權重已保存至 {save_dir}/model.msgpack")

if __name__ == "__main__":
    main()
