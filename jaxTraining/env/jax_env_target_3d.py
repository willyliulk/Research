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
    t_impact: float
    nominal_closing_speed: float
    
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

def get_uav_obs(state: EnvState, params: EnvParams) -> jnp.ndarray:
    dx = state.tar_pos[0] - state.uav_pos[0]
    dy = state.tar_pos[1] - state.uav_pos[1]
    dz = state.tar_pos[2] - state.uav_pos[2]
    R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    yaw_los = jnp.arctan2(dy, dx)
    pitch_los = jnp.arctan2(dz, jnp.hypot(dx, dy) + 1e-6)
    
    yaw_error = wrap_pi(yaw_los - state.uav_yaw)
    pitch_error = wrap_pi(pitch_los - state.uav_pitch)
    
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
    
    obs = jnp.array([
        jnp.clip(R / params.base_r, 0.0, 1.0),
        yaw_error / jnp.pi,
        pitch_error / (jnp.pi / 2),
        jnp.clip(state.uav_vel / params.max_vel, -1.0, 1.0),
        jnp.clip(yaw_los_rate / params.max_los_rate, -1.0, 1.0),
        jnp.clip(pitch_los_rate / params.max_los_rate, -1.0, 1.0),
        state.uav_pitch / (jnp.pi / 2)
    ], dtype=jnp.float32)
    
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    
    t_impact_norm = jnp.clip(state.t_impact / params.max_time, 0.0, 1.0)
    t_current_norm = jnp.clip(state.sim_time / params.max_time, 0.0, 1.0)
    
    t_ideal_remaining = state.t_impact - state.sim_time
    t_straight_remaining = R / state.nominal_closing_speed
    time_deficit = jnp.clip((t_ideal_remaining - t_straight_remaining) / params.max_time, -1.0, 1.0)
    
    time_feats = jnp.array([t_impact_norm, t_current_norm, time_deficit], dtype=jnp.float32)
    return jnp.concatenate([obs, time_feats])

def step_env(key, state: EnvState, action: jnp.ndarray, uav_action: jnp.ndarray, params: EnvParams) -> Tuple[Dict[str, jnp.ndarray], EnvState, float, bool, Dict]:
    # Action 是 Target 的控制 (範圍 -1 ~ 1)
    tar_act = jnp.clip(action, -1.0, 1.0)
    tar_cmd_pitch_accel = tar_act[0] * params.tar_max_pitch_accel
    tar_cmd_yaw_accel = tar_act[1] * params.tar_max_yaw_accel
    
    # UAV 透過神經網路提供的控制 (範圍 -1 ~ 1)
    uav_act = jnp.clip(uav_action, -1.0, 1.0)
    uav_cmd_pitch_accel = uav_act[0] * params.uav_max_pitch_accel
    uav_cmd_yaw_accel = uav_act[1] * params.uav_max_yaw_accel
    
    # ── 二階系統響應 (Tau = 0.707) ──
    tau = 0.707
    wn = 1.0 / tau
    zeta = 0.707
    
    # UAV 響應
    uav_pitch_accel_rate = state.uav_pitch_accel_rate + params.dt * (wn**2 * (uav_cmd_pitch_accel - state.uav_actual_pitch_accel) - 2 * zeta * wn * state.uav_pitch_accel_rate)
    uav_actual_pitch_accel = state.uav_actual_pitch_accel + params.dt * uav_pitch_accel_rate
    uav_yaw_accel_rate = state.uav_yaw_accel_rate + params.dt * (wn**2 * (uav_cmd_yaw_accel - state.uav_actual_yaw_accel) - 2 * zeta * wn * state.uav_yaw_accel_rate)
    uav_actual_yaw_accel = state.uav_actual_yaw_accel + params.dt * uav_yaw_accel_rate
    
    # Target 響應
    tar_pitch_accel_rate = state.tar_pitch_accel_rate + params.dt * (wn**2 * (tar_cmd_pitch_accel - state.tar_actual_pitch_accel) - 2 * zeta * wn * state.tar_pitch_accel_rate)
    tar_actual_pitch_accel = state.tar_actual_pitch_accel + params.dt * tar_pitch_accel_rate
    tar_yaw_accel_rate = state.tar_yaw_accel_rate + params.dt * (wn**2 * (tar_cmd_yaw_accel - state.tar_actual_yaw_accel) - 2 * zeta * wn * state.tar_yaw_accel_rate)
    tar_actual_yaw_accel = state.tar_actual_yaw_accel + params.dt * tar_yaw_accel_rate
    
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
    
    # Penalty if target goes too high or too low (keep it between 500 and 8000 ideally, but hard bound at 0)
    reward += jnp.where(next_tar_pos[2] < 500, -0.1, 0.0)
    reward += jnp.where(next_tar_pos[2] > 8000, -0.1, 0.0)
    
    # Terminations
    hit = R < 30.0
    timeout = sim_time >= params.max_time
    out_of_bounds = (R > params.base_r * 4.0) | (next_uav_pos[2] < 0.0) # UAV crashes or flies out
    tar_crash = next_tar_pos[2] < 0.0
    
    done = hit | timeout | out_of_bounds | tar_crash
    
    # Terminal Rewards
    reward += jnp.where(hit, -100.0, 0.0)         #被 UAV 擊落
    reward += jnp.where(tar_crash, -100.0, 0.0)   #目標自己墜海
    
    # 成功存活到最後或甩開 UAV
    success = timeout | out_of_bounds & (~tar_crash)
    reward += jnp.where(success, 100.0, 0.0)
    
    obs = get_obs(new_state, params)
    
    info = {
        "hit": hit,
        "timeout": timeout,
        "tar_crash": tar_crash,
        "out_of_bounds": out_of_bounds,
        "reward": reward,
        "episode_len": num_step
    }
    
    return obs, new_state, reward, done, info

def reset_env(key, params: EnvParams) -> Tuple[Dict[str, jnp.ndarray], EnvState]:
    keys = jax.random.split(key, 14)
    
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
    
    # ── 計算真實的閉合速度與物理極限攔截時間 ──
    dir_x = dx / (prev_R + 1e-6)
    dir_y = dy / (prev_R + 1e-6)
    dir_z = dz / (prev_R + 1e-6)
    
    tar_vx = tar_spd * jnp.cos(tar_pitch) * jnp.cos(tar_yaw)
    tar_vy = tar_spd * jnp.cos(tar_pitch) * jnp.sin(tar_yaw)
    tar_vz = tar_spd * jnp.sin(tar_pitch)
    
    tar_flee_speed = tar_vx * dir_x + tar_vy * dir_y + tar_vz * dir_z
    nominal_closing_speed = jnp.maximum(100.0, uav_spd - tar_flee_speed)
    
    t_min_intercept = prev_R / nominal_closing_speed
    
    # 隨機放大 1.1 ~ 1.5 倍
    random_offset = jax.random.uniform(keys[12], minval=1.1, maxval=1.5)
    t_impact = jnp.clip(t_min_intercept * random_offset, 0.0, params.max_time * 0.9)
    
    state = EnvState(
        uav_pos=uav_pos, uav_vel=uav_spd, uav_yaw=uav_yaw, uav_pitch=uav_pitch,
        uav_actual_pitch_accel=0.0, uav_actual_yaw_accel=0.0,
        uav_pitch_accel_rate=0.0, uav_yaw_accel_rate=0.0,
        tar_pos=tar_pos, tar_vel=tar_spd, tar_yaw=tar_yaw, tar_pitch=tar_pitch,
        tar_actual_pitch_accel=0.0, tar_actual_yaw_accel=0.0,
        tar_pitch_accel_rate=0.0, tar_yaw_accel_rate=0.0,
        prev_R=prev_R, sim_time=0.0, num_step=0,
        t_impact=t_impact, nominal_closing_speed=nominal_closing_speed,
        history_buffer=history_buffer
    )
    
    obs = get_obs(state, params)
    return obs, state
