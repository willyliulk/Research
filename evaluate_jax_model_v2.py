import jax
import jax.numpy as jnp
import flax
from jaxTraining.env.jax_env_3d import EnvParams, EnvState, reset_env, step_env
from jaxTraining.train_jax_ppo import ActorCritic
import numpy as np

def load_params(path):
    with open(path, "rb") as f:
        return flax.serialization.from_bytes(None, f.read())

def evaluate_optimized():
    try:
        params = load_params("jaxTraining/checkpoints/checkpoint_20260706-2340/model.msgpack")
    except:
        params = load_params("jaxTraining/checkpoints_v2/model.msgpack")
        
    network = ActorCritic(action_dim=2)
    env_params = EnvParams()
    
    key = jax.random.PRNGKey(42)
    num_envs = 1000
    init_steps = 3000
    
    # 1. 初始化環境
    keys = jax.random.split(key, num_envs)
    init_obsv, init_env_state = jax.vmap(reset_env, in_axes=(0, None))(keys, env_params)
    
    # 2. 定義 Scan 需要維護的 Carry 狀態
    # 我們需要追蹤：當前環境狀態、當前觀測、隨機金鑰、每個環境是否已經結束、以及要記錄的指標
    init_carry = {
        "env_state": init_env_state,
        "obsv": init_obsv,
        "key": key,
        "dones": jnp.zeros(num_envs, dtype=jnp.bool_),
        # 用來紀錄「第一次 Done」那一刻的數據
        "final_metrics": {
            "hit": jnp.zeros(num_envs, dtype=jnp.int32),
            "crash": jnp.zeros(num_envs, dtype=jnp.int32),
            "timeout": jnp.zeros(num_envs, dtype=jnp.int32),
            "time_err": jnp.zeros(num_envs, dtype=jnp.float32),
            "reward": jnp.zeros(num_envs, dtype=jnp.float32),
        }
    }
    
    vmap_step = jax.vmap(step_env, in_axes=(0, 0, 0, None))

    # 3. 定義單步的 Step 函數（完全純函數，無 Python 副作用）
    def scan_step(carry, _):
        current_key, step_key = jax.random.split(carry["key"])
        
        # 預測動作
        action_mean, action_logstd, value = network.apply(params, carry["obsv"])
        
        # 步進環境
        step_keys = jax.random.split(step_key, num_envs)
        next_obsv, next_env_state, reward, done, info = vmap_step(
            step_keys, carry["env_state"], action_mean, env_params
        )
        
        # 判定哪些環境是「在這一跨步首度結束」
        just_done = done & ~carry["dones"]
        
        # 如果首度結束，就把 info 裡面的數據填入 final_metrics；否則保持原樣
        def update_metric(current_val, info_val):
            return jnp.where(just_done, info_val, current_val)
        
        next_metrics = {
            "hit": update_metric(carry["final_metrics"]["hit"], info["hit"]),
            "crash": update_metric(carry["final_metrics"]["crash"], info["crash"]),
            "timeout": update_metric(carry["final_metrics"]["timeout"], info["timeout"]),
            "time_err": update_metric(carry["final_metrics"]["time_err"], info["time_err"]),
            "reward": update_metric(carry["final_metrics"]["reward"], info["reward"]),
        }
        
        next_carry = {
            "env_state": next_env_state,
            "obsv": next_obsv,
            "key": current_key,
            "dones": carry["dones"] | done,
            "final_metrics": next_metrics
        }
        
        # 我們只關心最後的 Carry，每一步不需要額外輸出 y，給 None 即可
        return next_carry, None

    # 4. 用 jax.jit 封裝整個 scan 循環！
    # 這行會把 1000 步的並行模擬直接編譯成一個極速硬體核心
    @jax.jit
    def run_rollout(c):
        final_c, _ = jax.lax.scan(scan_step, c, xs=None, length=init_steps)
        return final_c

    # 執行評估（這裡才會真正觸發 GPU/TPU 計算）
    final_carry = run_rollout(init_carry)
    metrics = final_carry["final_metrics"]
    
    # 5. 把最終統計結果拉回 CPU 列印
    print(f"Total evaluated: {num_envs}")
    print(f"Hits: {int(jnp.sum(metrics['hit']))} ({(jnp.sum(metrics['hit'])/num_envs)*100:.1f}%)")
    print(f"Crashes: {int(jnp.sum(metrics['crash']))} ({(jnp.sum(metrics['crash'])/num_envs)*100:.1f}%)")
    print(f"Timeouts: {int(jnp.sum(metrics['timeout']))} ({(jnp.sum(metrics['timeout'])/num_envs)*100:.1f}%)")
    print(f"Avg Time Error: {np.mean(metrics['time_err']):.3f} s")
    print(f"Avg Final Reward: {np.mean(metrics['reward']):.1f}")
    
    # 6. 將結果輸出成 CSV
    import pandas as pd
    
    init_uav_pos = np.array(jax.device_get(init_carry["env_state"].uav_pos))
    init_tar_pos = np.array(jax.device_get(init_carry["env_state"].tar_pos))
    init_uav_vel = np.array(jax.device_get(init_carry["env_state"].uav_vel))
    init_uav_pitch = np.array(jax.device_get(init_carry["env_state"].uav_pitch))
    init_uav_yaw = np.array(jax.device_get(init_carry["env_state"].uav_yaw))
    init_tar_vel = np.array(jax.device_get(init_carry["env_state"].tar_vel))
    init_tar_pitch = np.array(jax.device_get(init_carry["env_state"].tar_pitch))
    init_tar_yaw = np.array(jax.device_get(init_carry["env_state"].tar_yaw))
    init_t_impact = np.array(jax.device_get(init_carry["env_state"].t_impact))
    init_nominal_spd = np.array(jax.device_get(init_carry["env_state"].nominal_closing_speed))
    
    # 計算笛卡爾座標下的初始速度向量 (vx, vy, vz)
    init_uav_vx = init_uav_vel * np.cos(init_uav_pitch) * np.cos(init_uav_yaw)
    init_uav_vy = init_uav_vel * np.cos(init_uav_pitch) * np.sin(init_uav_yaw)
    init_uav_vz = init_uav_vel * np.sin(init_uav_pitch)
    
    init_tar_vx = init_tar_vel * np.cos(init_tar_pitch) * np.cos(init_tar_yaw)
    init_tar_vy = init_tar_vel * np.cos(init_tar_pitch) * np.sin(init_tar_yaw)
    init_tar_vz = init_tar_vel * np.sin(init_tar_pitch)
    
    final_hits = np.array(jax.device_get(metrics['hit']))
    final_crashes = np.array(jax.device_get(metrics['crash']))
    final_timeouts = np.array(jax.device_get(metrics['timeout']))
    final_time_err = np.array(jax.device_get(metrics['time_err']))
    final_reward = np.array(jax.device_get(metrics['reward']))
    
    status_list = []
    for h, c, t in zip(final_hits, final_crashes, final_timeouts):
        if h: status_list.append("Hit")
        elif c: status_list.append("Crash")
        elif t: status_list.append("Timeout")
        else: status_list.append("Out of Bounds")
        
    df = pd.DataFrame({
        "init_uav_x": init_uav_pos[:, 0],
        "init_uav_y": init_uav_pos[:, 1],
        "init_uav_z": init_uav_pos[:, 2],
        "init_tar_x": init_tar_pos[:, 0],
        "init_tar_y": init_tar_pos[:, 1],
        "init_tar_z": init_tar_pos[:, 2],
        "init_uav_vx": init_uav_vx,
        "init_uav_vy": init_uav_vy,
        "init_uav_vz": init_uav_vz,
        "init_tar_vx": init_tar_vx,
        "init_tar_vy": init_tar_vy,
        "init_tar_vz": init_tar_vz,
        "nominal_closing_speed": init_nominal_spd,
        "t_impact_target": init_t_impact,
        "status": status_list,
        "time_error": final_time_err,
        "reward": final_reward
    })
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = f"evaluation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 評估結果已儲存至: {csv_path}")

if __name__ == "__main__":
    evaluate_optimized()