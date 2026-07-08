import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict

@struct.dataclass
class EnvState:
    uav_pos: jnp.ndarray      # [3]
    uav_vel: float
    uav_yaw: float
    uav_pitch: float
    
    uav_actual_pitch_accel: float
    uav_actual_yaw_accel: float
    uav_pitch_accel_rate: float
    uav_yaw_accel_rate: float
    
    tar_pos: jnp.ndarray      # [3]
    tar_vel: float
    tar_yaw: float
    tar_pitch: float
    
    tar_actual_pitch_accel: float
    tar_actual_yaw_accel: float
    tar_pitch_accel_rate: float
    tar_yaw_accel_rate: float

    prev_R: float
    sim_time: float
    num_step: int
    
    # History Buffer (Shape: [25, 3] -> R, LOS_yaw, LOS_pitch)
    history_buffer: jnp.ndarray

@struct.dataclass
class EnvParams:
    dt: float = 0.016
    uav_max_pitch_accel: float = 3000.0
    uav_max_yaw_accel: float = 3000.0
    tar_max_pitch_accel: float = 6000.0
    tar_max_yaw_accel: float = 6000.0
    max_steps: int = 3000
    base_r: float = 8000.0
    max_vel: float = 2500.0
    max_los_rate: float = 0.3
    max_time: float = 30.0
    
    history_len: int = 25
    history_interval: int = 12 # 12 * 0.016 = 0.192s
    
    N_png: float = 3.0 # PNG Navigation Constant
    pp_dist_threshold: float = 500.0 # Switch to Pure Pursuit when closer than 500m
    K_pp: float = 20.0 # Pure Pursuit Proportional Gain

def wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def get_obs(state: EnvState, params: EnvParams) -> Dict[str, jnp.ndarray]:
    dx = state.tar_pos[0] - state.uav_pos[0]
    dy = state.tar_pos[1] - state.uav_pos[1]
    dz = state.tar_pos[2] - state.uav_pos[2]
    R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    # 從 Target 視角看的 LOS
    yaw_los = jnp.arctan2(-dy, -dx) # Target to UAV
    pitch_los = jnp.arctan2(-dz, jnp.hypot(-dx, -dy) + 1e-6)
    
    yaw_error = wrap_pi(yaw_los - state.tar_yaw)
    pitch_error = wrap_pi(pitch_los - state.tar_pitch)
    
    # 建立 Current Observation (MLP input)
    obs_current = jnp.array([
        jnp.clip(R / params.base_r, 0.0, 1.0),
        yaw_error / jnp.pi,
        pitch_error / (jnp.pi / 2),
        jnp.clip(state.tar_vel / params.max_vel, -1.0, 1.0),
        jnp.clip(state.uav_vel / params.max_vel, -1.0, 1.0),
        state.tar_pitch / (jnp.pi / 2)
    ], dtype=jnp.float32)
    obs_current = jnp.nan_to_num(obs_current, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # History Buffer (Transformer input)
    # Shape is already [25, 3] from state
    # We normalize it for the network
    # The features are R, LOS_yaw, LOS_pitch (Relative to Target!)
    
    # Create normalized history buffer
    R_hist = jnp.clip(state.history_buffer[:, 0] / params.base_r, 0.0, 1.0)
    yaw_hist = state.history_buffer[:, 1] / jnp.pi
    pitch_hist = state.history_buffer[:, 2] / (jnp.pi / 2)
    
    obs_history = jnp.stack([R_hist, yaw_hist, pitch_hist], axis=-1)
    
    return {
        "current": obs_current,
        "history": obs_history
    }

def get_uav_guidance_accel(state: EnvState, params: EnvParams) -> Tuple[float, float]:
    """計算 UAV 的硬編碼導引律加速度 (PNG + PP)"""
    dx = state.tar_pos[0] - state.uav_pos[0]
    dy = state.tar_pos[1] - state.uav_pos[1]
    dz = state.tar_pos[2] - state.uav_pos[2]
    R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    # UAV to Target LOS
    yaw_los = jnp.arctan2(dy, dx)
    pitch_los = jnp.arctan2(dz, jnp.hypot(dx, dy) + 1e-6)
    
    yaw_err = wrap_pi(yaw_los - state.uav_yaw)
    pitch_err = wrap_pi(pitch_los - state.uav_pitch)
    
    # 速度向量
    ax_v = state.uav_vel * jnp.cos(state.uav_pitch) * jnp.cos(state.uav_yaw)
    ay_v = state.uav_vel * jnp.cos(state.uav_pitch) * jnp.sin(state.uav_yaw)
    az_v = state.uav_vel * jnp.sin(state.uav_pitch)
    
    bx_v = state.tar_vel * jnp.cos(state.tar_pitch) * jnp.cos(state.tar_yaw)
    by_v = state.tar_vel * jnp.cos(state.tar_pitch) * jnp.sin(state.tar_yaw)
    bz_v = state.tar_vel * jnp.sin(state.tar_pitch)
    
    dvx = bx_v - ax_v
    dvy = by_v - ay_v
    dvz = bz_v - az_v
    
    r_xy = jnp.hypot(dx, dy)
    yaw_los_rate = (dx * dvy - dy * dvx) / (r_xy**2 + 1e-6)
    d_r_xy = (dx * dvx + dy * dvy) / (r_xy + 1e-6)
    pitch_los_rate = (r_xy * dvz - dz * d_r_xy) / (R**2 + 1e-6)
    
    closing_vel = -(dx*dvx + dy*dvy + dz*dvz) / (R + 1e-6)
    
    # PNG: a = N * V_c * LOS_rate
    png_yaw_accel = params.N_png * closing_vel * yaw_los_rate
    png_pitch_accel = params.N_png * closing_vel * pitch_los_rate
    
    # PP: a = K * Error
    pp_yaw_accel = params.K_pp * yaw_err
    pp_pitch_accel = params.K_pp * pitch_err
    
    # 判斷是否切換為 PP
    use_pp = R < params.pp_dist_threshold
    
    cmd_yaw_accel = jnp.where(use_pp, pp_yaw_accel, png_yaw_accel)
    cmd_pitch_accel = jnp.where(use_pp, pp_pitch_accel, png_pitch_accel)
    
    # Clip to max
    cmd_yaw_accel = jnp.clip(cmd_yaw_accel, -params.uav_max_yaw_accel, params.uav_max_yaw_accel)
    cmd_pitch_accel = jnp.clip(cmd_pitch_accel, -params.uav_max_pitch_accel, params.uav_max_pitch_accel)
    
    return cmd_pitch_accel, cmd_yaw_accel

def get_uav_obs(state: EnvState, params: EnvParams) -> jnp.ndarray:
    # 計算 UAV 推論模型的 10 維觀測向量 (dvx, dvy, dvz, dx, dy, dz, los_rate_y, los_rate_z, t_impact_norm, time_deficit)
    dx = state.tar_pos[0] - state.uav_pos[0]
    dy = state.tar_pos[1] - state.uav_pos[1]
    dz = state.tar_pos[2] - state.uav_pos[2]
    
    # 笛卡爾速度差
    uav_vx = state.uav_vel * jnp.cos(state.uav_pitch) * jnp.cos(state.uav_yaw)
    uav_vy = state.uav_vel * jnp.cos(state.uav_pitch) * jnp.sin(state.uav_yaw)
    uav_vz = state.uav_vel * jnp.sin(state.uav_pitch)
    tar_vx = state.tar_vel * jnp.cos(state.tar_pitch) * jnp.cos(state.tar_yaw)
    tar_vy = state.tar_vel * jnp.cos(state.tar_pitch) * jnp.sin(state.tar_yaw)
    tar_vz = state.tar_vel * jnp.sin(state.tar_pitch)
    
    dvx = tar_vx - uav_vx
    dvy = tar_vy - uav_vy
    dvz = tar_vz - uav_vz
    
    R = jnp.hypot(jnp.hypot(dx, dy), dz)
    v_close = -(dx*dvx + dy*dvy + dz*dvz) / (R + 1e-6)
    
    # Cross product for LOS rate vector (rel_pos x rel_vel)
    cross_x = dy*dvz - dz*dvy
    cross_y = dz*dvx - dx*dvz
    cross_z = dx*dvy - dy*dvx
    
    los_rate_y = cross_y / (R**2 + 1e-6)
    los_rate_z = cross_z / (R**2 + 1e-6)
    
    t_impact = jnp.clip(R / (v_close + 1e-6), 0.0, 30.0)
    t_impact_norm = jnp.clip(t_impact / 30.0, -1.0, 1.0)
    time_deficit = jnp.clip((params.max_time - state.sim_time)/params.max_time, 0.0, 1.0)
    
    obs = jnp.array([
        dvx/1000.0, dvy/1000.0, dvz/1000.0,
        dx/10000.0, dy/10000.0, dz/10000.0,
        jnp.clip(los_rate_y * 10.0, -1.0, 1.0),
        jnp.clip(los_rate_z * 10.0, -1.0, 1.0),
        t_impact_norm,
        time_deficit
    ], dtype=jnp.float32)
    return obs

def step_env(key, state: EnvState, action: jnp.ndarray, arg4, arg5=None):
    if arg5 is None:
        uav_action = None
        params = arg4
    else:
        uav_action = arg4
        params = arg5
        
    # Action 是 Target 的控制 (範圍 -1 ~ 1)
    tar_act = jnp.clip(action, -1.0, 1.0)
    tar_cmd_pitch_accel = tar_act[0] * params.tar_max_pitch_accel
    tar_cmd_yaw_accel = tar_act[1] * params.tar_max_yaw_accel
    
    if uav_action is None:
        # UAV 硬編碼控制
        uav_cmd_pitch_accel, uav_cmd_yaw_accel = get_uav_guidance_accel(state, params)
    else:
        # 神經網路控制 (範圍 -1 ~ 1)
        uav_act = jnp.clip(uav_action, -1.0, 1.0)
        uav_cmd_pitch_accel = uav_act[0] * params.uav_max_pitch_accel
        uav_cmd_yaw_accel = uav_act[1] * params.uav_max_yaw_accel

    
    # ── 二階系統響應 (Tau = 0.707) ──
    tau = 0.707
    wn = 1.0 / tau
    zeta = 0.707
    
    # UAV 響應 (拔除二階延遲，改為一階瞬間發力，以對齊預訓練 UAV 模型與 Pygame 模擬環境)
    uav_pitch_accel_rate = 0.0
    uav_actual_pitch_accel = uav_cmd_pitch_accel
    uav_yaw_accel_rate = 0.0
    uav_actual_yaw_accel = uav_cmd_yaw_accel
    
    # Target 響應 (拔除二階延遲，改為一階瞬間發力，以實現完全公平的 0 延遲狗鬥)
    tar_pitch_accel_rate = 0.0
    tar_actual_pitch_accel = tar_cmd_pitch_accel
    tar_yaw_accel_rate = 0.0
    tar_actual_yaw_accel = tar_cmd_yaw_accel
    
    # ── 運動學積分 ──
    # UAV
    next_uav_pitch = jnp.clip(state.uav_pitch + params.dt * uav_actual_pitch_accel / state.uav_vel, -jnp.pi/2 * 0.99, jnp.pi/2 * 0.99)
    next_uav_yaw = wrap_pi(state.uav_yaw + params.dt * uav_actual_yaw_accel / (state.uav_vel * jnp.cos(state.uav_pitch) + 1e-6))
    ax_v = state.uav_vel * jnp.cos(next_uav_pitch) * jnp.cos(next_uav_yaw)
    ay_v = state.uav_vel * jnp.cos(next_uav_pitch) * jnp.sin(next_uav_yaw)
    az_v = state.uav_vel * jnp.sin(next_uav_pitch)
    next_uav_pos = state.uav_pos + params.dt * jnp.array([ax_v, ay_v, az_v])
    
    # Target
    next_tar_pitch = jnp.clip(state.tar_pitch + params.dt * tar_actual_pitch_accel / state.tar_vel, -jnp.pi/2 * 0.99, jnp.pi/2 * 0.99)
    next_tar_yaw = wrap_pi(state.tar_yaw + params.dt * tar_actual_yaw_accel / (state.tar_vel * jnp.cos(state.tar_pitch) + 1e-6))
    bx_v = state.tar_vel * jnp.cos(next_tar_pitch) * jnp.cos(next_tar_yaw)
    by_v = state.tar_vel * jnp.cos(next_tar_pitch) * jnp.sin(next_tar_yaw)
    bz_v = state.tar_vel * jnp.sin(next_tar_pitch)
    next_tar_pos = state.tar_pos + params.dt * jnp.array([bx_v, by_v, bz_v])
    
    dx = next_tar_pos[0] - next_uav_pos[0]
    dy = next_tar_pos[1] - next_uav_pos[1]
    dz = next_tar_pos[2] - next_uav_pos[2]
    R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    # ── History Buffer 更新 ──
    num_step = state.num_step + 1
    sim_time = state.sim_time + params.dt
    
    # 從 Target 看 UAV 的角度
    tar_yaw_los = jnp.arctan2(-dy, -dx)
    tar_pitch_los = jnp.arctan2(-dz, jnp.hypot(-dx, -dy) + 1e-6)
    tar_yaw_err = wrap_pi(tar_yaw_los - next_tar_yaw)
    tar_pitch_err = wrap_pi(tar_pitch_los - next_tar_pitch)
    
    current_feat = jnp.array([R, tar_yaw_err, tar_pitch_err])
    
    def update_buffer(buffer):
        return jnp.concatenate([buffer[1:], current_feat[None, :]], axis=0)
    
    next_history_buffer = jax.lax.cond(
        num_step % params.history_interval == 0,
        update_buffer,
        lambda b: b,
        state.history_buffer
    )
    
    new_state = state.replace(
        uav_pos=next_uav_pos, uav_pitch=next_uav_pitch, uav_yaw=next_uav_yaw,
        uav_actual_pitch_accel=uav_actual_pitch_accel, uav_actual_yaw_accel=uav_actual_yaw_accel,
        uav_pitch_accel_rate=uav_pitch_accel_rate, uav_yaw_accel_rate=uav_yaw_accel_rate,
        tar_pos=next_tar_pos, tar_pitch=next_tar_pitch, tar_yaw=next_tar_yaw,
        tar_actual_pitch_accel=tar_actual_pitch_accel, tar_actual_yaw_accel=tar_actual_yaw_accel,
        tar_pitch_accel_rate=tar_pitch_accel_rate, tar_yaw_accel_rate=tar_yaw_accel_rate,
        prev_R=R, sim_time=sim_time, num_step=num_step,
        history_buffer=next_history_buffer
    )
    
    # ── Reward Design (Target Evasion) ──
    reward = 0.0
    
    # Survival bonus per step
    reward += 0.05
    
    # Fuel/Control Penalty (Target's objective is to minimize fuel usage, i.e., use less accel)
    fuel_penalty = -0.01 * (jnp.abs(tar_act[0]) + jnp.abs(tar_act[1]))
    reward += fuel_penalty
    
    # Penalty if target goes too high or too low
    reward += jnp.where(next_tar_pos[2] < 1500, -0.5, 0.0) # 強烈懲罰低飛 (防止往地板鑽)
    reward += jnp.where(next_tar_pos[2] > 9000, -0.1, 0.0)
    
    # Target 自己的出界判定 (限制戰鬥區域)
    tar_oob = (jnp.abs(next_tar_pos[0]) > 10000.0) | (jnp.abs(next_tar_pos[1]) > 10000.0) | (next_tar_pos[2] > 10000.0)
    
    # Terminations
    hit = R < 30.0
    timeout = sim_time >= params.max_time
    uav_crash = next_uav_pos[2] < 0.0
    tar_crash = next_tar_pos[2] < 0.0
    
    done = hit | timeout | tar_oob | uav_crash | tar_crash
    
    # Terminal Rewards
    reward += jnp.where(hit, -100.0, 0.0)         # 被 UAV 擊落
    reward += jnp.where(tar_crash, -100.0, 0.0)   # 目標自己墜海
    reward += jnp.where(tar_oob, -100.0, 0.0)     # 目標逃離戰區 (消極避戰)
    
    # 成功存活到最後或成功誘導 UAV 墜毀 (且自己沒事)
    success = (timeout | uav_crash) & (~tar_crash) & (~tar_oob) & (~hit)
    reward += jnp.where(success, 100.0, 0.0)
    
    obs = get_obs(new_state, params)
    
    info = {
        "hit": hit,
        "timeout": timeout,
        "tar_crash": tar_crash,
        "out_of_bounds": tar_oob,
        "reward": reward,
        "episode_len": num_step
    }
    
    return obs, new_state, reward, done, info

def reset_env(key, params: EnvParams) -> Tuple[Dict[str, jnp.ndarray], EnvState]:
    keys = jax.random.split(key, 12)
    
    tar_spd = jax.random.uniform(keys[0], minval=300.0, maxval=800.0)
    uav_spd = tar_spd + jax.random.uniform(keys[1], minval=600.0, maxval=1200.0) # UAV much faster
    
    # Random starting positions
    tar_pos = jnp.array([
        jax.random.uniform(keys[2], minval=-3000.0, maxval=3000.0),
        jax.random.uniform(keys[3], minval=-3000.0, maxval=3000.0),
        jax.random.uniform(keys[4], minval=3000.0, maxval=8000.0)
    ])
    
    init_dist = jax.random.uniform(keys[5], minval=2000.0, maxval=6000.0)
    init_azimuth = jax.random.uniform(keys[6], minval=-jnp.pi, maxval=jnp.pi)
    init_elevation = jax.random.uniform(keys[7], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    uav_pos = tar_pos + init_dist * jnp.array([
        jnp.cos(init_elevation) * jnp.cos(init_azimuth),
        jnp.cos(init_elevation) * jnp.sin(init_azimuth),
        jnp.sin(init_elevation)
    ])
    uav_pos = uav_pos.at[2].set(jnp.maximum(1000.0, uav_pos[2]))
    
    # Head-on geometry by default to make it hard
    dx = tar_pos[0] - uav_pos[0]
    dy = tar_pos[1] - uav_pos[1]
    dz = tar_pos[2] - uav_pos[2]
    
    uav_yaw = jnp.arctan2(dy, dx) + jax.random.uniform(keys[8], minval=-jnp.pi/4, maxval=jnp.pi/4)
    uav_pitch = jnp.arctan2(dz, jnp.hypot(dx, dy) + 1e-6) + jax.random.uniform(keys[9], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    tar_yaw = wrap_pi(uav_yaw + jnp.pi) + jax.random.uniform(keys[10], minval=-jnp.pi/4, maxval=jnp.pi/4)
    tar_pitch = jax.random.uniform(keys[11], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    prev_R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    # Initialize history buffer with the starting conditions
    tar_yaw_los = jnp.arctan2(-dy, -dx)
    tar_pitch_los = jnp.arctan2(-dz, jnp.hypot(-dx, -dy) + 1e-6)
    tar_yaw_err = wrap_pi(tar_yaw_los - tar_yaw)
    tar_pitch_err = wrap_pi(tar_pitch_los - tar_pitch)
    init_feat = jnp.array([prev_R, tar_yaw_err, tar_pitch_err])
    history_buffer = jnp.repeat(init_feat[None, :], 25, axis=0)
    
    state = EnvState(
        uav_pos=uav_pos, uav_vel=uav_spd, uav_yaw=uav_yaw, uav_pitch=uav_pitch,
        uav_actual_pitch_accel=0.0, uav_actual_yaw_accel=0.0,
        uav_pitch_accel_rate=0.0, uav_yaw_accel_rate=0.0,
        tar_pos=tar_pos, tar_vel=tar_spd, tar_yaw=tar_yaw, tar_pitch=tar_pitch,
        tar_actual_pitch_accel=0.0, tar_actual_yaw_accel=0.0,
        tar_pitch_accel_rate=0.0, tar_yaw_accel_rate=0.0,
        prev_R=prev_R, sim_time=0.0, num_step=0,
        history_buffer=history_buffer
    )
    
    obs = get_obs(state, params)
    return obs, state
