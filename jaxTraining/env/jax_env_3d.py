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

    prev_R: float
    sim_time: float
    num_step: int
    
    t_impact: float
    episode_target_motion: int
    
    t0: float
    prev_motion_type: int
    nominal_closing_speed: float

@struct.dataclass
class EnvParams:
    dt: float = 0.016
    uav_max_pitch_accel: float = 5000.0
    uav_max_yaw_accel: float = 5000.0
    tar_max_pitch_accel: float = 8000.0
    tar_max_yaw_accel: float = 8000.0
    w_dist: float = 0.0
    w_time: float = 30.0
    los_penalty_scale: float = 0.05
    max_steps: int = 3000
    base_r: float = 8000.0
    max_vel: float = 1500.0
    max_los_rate: float = 0.3
    max_time: float = 30.0
    target_spd_vray_rate: float = 0.01
    obs_noise_std: float = 0.0   # 關閉觀測噪聲，回到乾淨 Baseline
    use_2nd_order: bool = False  # 關閉二階系統測試
    aw_yaw: float = 4000.0     # TAR_MAX_YAW_ACCEL * 0.5
    aw_pitch: float = 4000.0   # TAR_MAX_PITCH_ACCEL * 0.5
    Ww: float = 4.0
    phiw: float = 0.0

def wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def get_target_accel(state: EnvState, params: EnvParams) -> Tuple[float, float, EnvState]:
    motion_type = state.episode_target_motion
    
    changed = motion_type != state.prev_motion_type
    t0 = jnp.where(changed, state.sim_time, state.t0)
    prev_motion_type = motion_type
    
    t = state.sim_time - t0
    
    def f_nothing(): return 0.0, 0.0
    def f_pitch_up(): return params.tar_max_pitch_accel, 0.0
    def f_pitch_down(): return -params.tar_max_pitch_accel, 0.0
    def f_yaw_left(): return 0.0, params.tar_max_yaw_accel * 0.3
    def f_yaw_right(): return 0.0, -params.tar_max_yaw_accel * 0.3
    def f_weaving(): return 0.0, params.aw_yaw * jnp.sin(params.Ww * t + params.phiw)
    def f_spiral(): return params.aw_pitch * jnp.cos(params.Ww * t), params.aw_yaw * jnp.sin(params.Ww * t)

    ap, al = jax.lax.switch(
        motion_type,
        [f_nothing, f_pitch_up, f_pitch_down, f_yaw_left, f_yaw_right, f_weaving, f_spiral]
    )

    new_state = state.replace(t0=t0, prev_motion_type=prev_motion_type)
    return ap, al, new_state

def get_obs(key, state: EnvState, params: EnvParams) -> jnp.ndarray:
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
    
    noise = jax.random.normal(key, shape=obs.shape) * params.obs_noise_std
    obs = jnp.where(params.obs_noise_std > 0.0, jnp.clip(obs + noise, -1.0, 1.0), obs)
    
    t_impact_norm = jnp.clip(state.t_impact / params.max_time, 0.0, 1.0)
    t_current_norm = jnp.clip(state.sim_time / params.max_time, 0.0, 1.0)
    
    t_ideal_remaining = state.t_impact - state.sim_time
    t_straight_remaining = R / state.nominal_closing_speed
    time_deficit = jnp.clip((t_ideal_remaining - t_straight_remaining) / params.max_time, -1.0, 1.0)
    
    time_feats = jnp.array([t_impact_norm, t_current_norm, time_deficit], dtype=jnp.float32)
    return jnp.concatenate([obs, time_feats])

def step_env(key, state: EnvState, action: jnp.ndarray, params: EnvParams) -> Tuple[jnp.ndarray, EnvState, float, bool, Dict]:
    uav_act = jnp.clip(action, -1.0, 1.0)
    cmd_pitch_accel = uav_act[0] * params.uav_max_pitch_accel
    cmd_yaw_accel = uav_act[1] * params.uav_max_yaw_accel
    
    # ── Autopilot 二階系統 ──────────────────────────────────────────────
    tau = 0.707
    wn = 1.0 / tau
    zeta = 0.707
    
    uav_pitch_accel_rate = state.uav_pitch_accel_rate + params.dt * (wn**2 * (cmd_pitch_accel - state.uav_actual_pitch_accel) - 2 * zeta * wn * state.uav_pitch_accel_rate)
    uav_actual_pitch_accel = state.uav_actual_pitch_accel + params.dt * uav_pitch_accel_rate
    
    uav_yaw_accel_rate = state.uav_yaw_accel_rate + params.dt * (wn**2 * (cmd_yaw_accel - state.uav_actual_yaw_accel) - 2 * zeta * wn * state.uav_yaw_accel_rate)
    uav_actual_yaw_accel = state.uav_actual_yaw_accel + params.dt * uav_yaw_accel_rate
    
    # 根據開關決定是否使用二階延遲
    uav_pitch_accel = jnp.where(params.use_2nd_order, uav_actual_pitch_accel, cmd_pitch_accel)
    uav_yaw_accel = jnp.where(params.use_2nd_order, uav_actual_yaw_accel, cmd_yaw_accel)
    uav_actual_pitch_accel = jnp.where(params.use_2nd_order, uav_actual_pitch_accel, cmd_pitch_accel)
    uav_actual_yaw_accel = jnp.where(params.use_2nd_order, uav_actual_yaw_accel, cmd_yaw_accel)
    
    target_pitch_accel, target_yaw_accel, state = get_target_accel(state, params)
    
    uav_dot_pitch = uav_pitch_accel / state.uav_vel
    uav_cos_pitch_safe = jnp.maximum(0.01, jnp.cos(state.uav_pitch))
    uav_dot_yaw = uav_yaw_accel / (state.uav_vel * uav_cos_pitch_safe)
    
    uav_pitch = jnp.clip(state.uav_pitch + uav_dot_pitch * params.dt, -jnp.pi/2 + 0.01, jnp.pi/2 - 0.01)
    uav_yaw = wrap_pi(state.uav_yaw + uav_dot_yaw * params.dt)
    
    uav_pos = state.uav_pos + state.uav_vel * params.dt * jnp.array([
        jnp.cos(uav_pitch) * jnp.cos(uav_yaw),
        jnp.cos(uav_pitch) * jnp.sin(uav_yaw),
        jnp.sin(uav_pitch)
    ])
    
    tar_dot_pitch = target_pitch_accel / state.tar_vel
    tar_cos_pitch_safe = jnp.maximum(0.01, jnp.cos(state.tar_pitch))
    tar_dot_yaw = target_yaw_accel / (state.tar_vel * tar_cos_pitch_safe)
    
    tar_pitch = jnp.clip(state.tar_pitch + tar_dot_pitch * params.dt, -jnp.pi/2 + 0.01, jnp.pi/2 - 0.01)
    tar_yaw = wrap_pi(state.tar_yaw + tar_dot_yaw * params.dt)
    
    tar_vel = state.tar_vel
    key, subkey1, subkey2 = jax.random.split(key, 3)
    do_vray = jax.random.uniform(subkey1) < params.target_spd_vray_rate
    vel_noise = jax.random.uniform(subkey2, minval=tar_vel*-0.4, maxval=tar_vel*0.4)
    tar_vel = jnp.where(do_vray, tar_vel + vel_noise, tar_vel)
    
    tar_pos = state.tar_pos + tar_vel * params.dt * jnp.array([
        jnp.cos(tar_pitch) * jnp.cos(tar_yaw),
        jnp.cos(tar_pitch) * jnp.sin(tar_yaw),
        jnp.sin(tar_pitch)
    ])
    
    dx = tar_pos[0] - uav_pos[0]
    dy = tar_pos[1] - uav_pos[1]
    dz = tar_pos[2] - uav_pos[2]
    new_R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    closing_rate = (state.prev_R - new_R) / params.dt
    r_dist = params.w_dist * jnp.clip(closing_rate, -3000.0, 3000.0)
    
    num_step = state.num_step + 1
    sim_time = state.sim_time + params.dt
    
    # 恢復傳統的穩定 Reward，避免時間控制與靠近目標發生嚴重衝突
    # 距離懲罰 (越遠扣越多)
    dist_penalty = -0.01 * (new_R / params.base_r)
    # 計算速度分量以求 LOS rate
    ax_v = state.uav_vel * jnp.cos(uav_pitch) * jnp.cos(uav_yaw)
    ay_v = state.uav_vel * jnp.cos(uav_pitch) * jnp.sin(uav_yaw)
    az_v = state.uav_vel * jnp.sin(uav_pitch)
    
    bx_v = tar_vel * jnp.cos(tar_pitch) * jnp.cos(tar_yaw)
    by_v = tar_vel * jnp.cos(tar_pitch) * jnp.sin(tar_yaw)
    bz_v = tar_vel * jnp.sin(tar_pitch)
    
    dvx = bx_v - ax_v
    dvy = by_v - ay_v
    dvz = bz_v - az_v
    
    r_xy = jnp.hypot(dx, dy)
    yaw_los_rate = (dx * dvy - dy * dvx) / (r_xy**2 + 1e-6)
    d_r_xy = (dx * dvx + dy * dvy) / (r_xy + 1e-6)
    pitch_los_rate = (r_xy * dvz - dz * d_r_xy) / (new_R**2 + 1e-6)
    yaw_los = jnp.arctan2(dy, dx)
    pitch_los = jnp.arctan2(dz, r_xy + 1e-6)
    
    # 視線旋轉率懲罰 (LOS rate 越大扣越多，鼓勵直指目標)
    los_rate = jnp.sqrt(yaw_los_rate**2 + pitch_los_rate**2)
    
    # ── 時間誤差勢能獎勵 (Dense Time Reward) ──
    prev_ideal_remaining = state.t_impact - state.sim_time
    prev_straight_remaining = state.prev_R / state.nominal_closing_speed
    prev_time_deficit = prev_ideal_remaining - prev_straight_remaining
    
    new_ideal_remaining = state.t_impact - sim_time
    new_straight_remaining = new_R / state.nominal_closing_speed
    new_time_deficit = new_ideal_remaining - new_straight_remaining
    
    # 理論上正確的 Ng Potential-based Reward Shaping: F(s, s') = gamma * Phi(s') - Phi(s)
    # Phi(s) = -w_time * |time_deficit|
    # PPO 使用 gamma = 0.9995，所以 shaping 必須包含 gamma 否則會產生無限農分漏洞 (Reward Hacking)
    gamma = 0.9995
    r_time = params.w_time * (jnp.abs(prev_time_deficit) - gamma * jnp.abs(new_time_deficit))
    
    # ── 視線角懲罰 (LOS Penalty) ──
    # 移除原本的動態 multiplier，因為它會在 time_deficit = 0 處產生巨大的非連續斷層 (Cliff)
    # 這會嚇得模型不敢把 time_deficit 降到 0。
    # 改為極小的常數懲罰，僅作為軌跡平滑的正則化 (Regularization)
    los_penalty = -params.los_penalty_scale * (los_rate / params.max_los_rate)
    
    # 加入 r_dist 獎勵接近速度，加入 r_time 獎勵時間控制
    reward = dist_penalty + los_penalty + r_dist + r_time
    
    hit = new_R < 30.0
    over = num_step >= params.max_steps
    out_of_bound = new_R > (params.base_r * 4.0)
    uav_crash = uav_pos[2] < 0.0
    tar_crash = tar_pos[2] < 0.0
    
    time_err = jnp.abs(sim_time - state.t_impact)
    
    def on_hit():
        # 強化打擊時間的結算獎勵與懲罰
        # 引入有界的指數型時間獎勵 (Bounded Terminal Time Reward)
        time_bonus = 500.0 * jnp.exp(-0.5 * time_err)
        return reward + 1000.0 + time_bonus
    def on_uav_crash():
        return reward - 5000.0
    def on_out_of_bound():
        return reward - 5000.0
    def on_over():
        return reward - 20.0
    def on_ongoing():
        return reward

    final_reward = jnp.where(hit, on_hit(),
                   jnp.where(uav_crash, on_uav_crash(),
                   jnp.where(out_of_bound, on_out_of_bound(),
                   jnp.where(tar_crash | over, on_over(), on_ongoing()))))
                   
    done = hit | uav_crash | tar_crash | over | out_of_bound
    
    new_state = state.replace(
        uav_pos=uav_pos, uav_vel=state.uav_vel, uav_yaw=uav_yaw, uav_pitch=uav_pitch,
        uav_actual_pitch_accel=uav_actual_pitch_accel, uav_actual_yaw_accel=uav_actual_yaw_accel,
        uav_pitch_accel_rate=uav_pitch_accel_rate, uav_yaw_accel_rate=uav_yaw_accel_rate,
        tar_pos=tar_pos, tar_vel=tar_vel, tar_yaw=tar_yaw, tar_pitch=tar_pitch,
        prev_R=new_R, sim_time=sim_time, num_step=num_step
    )
    
    key, obs_key = jax.random.split(key)
    obs = get_obs(obs_key, new_state, params)
    
    info = {
        "hit": hit,
        "crash": uav_crash,
        "timeout": over | out_of_bound,
        "time_err": time_err,
        "episode_len": num_step,
        "reward": final_reward
    }
    
    return obs, new_state, final_reward, done, info

def reset_env(key, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
    keys = jax.random.split(key, 16)
    
    tar_spd = jax.random.uniform(keys[0], minval=600.0, maxval=1500.0)
    uav_spd = tar_spd + jax.random.uniform(keys[1], minval=500.0, maxval=1000.0)
    
    uav_pos = jnp.array([
        jax.random.uniform(keys[2], minval=-3000.0, maxval=3000.0),
        jax.random.uniform(keys[3], minval=-3000.0, maxval=3000.0),
        jax.random.uniform(keys[4], minval=3000.0, maxval=8000.0)
    ])
    uav_yaw = jax.random.uniform(keys[5], minval=-jnp.pi, maxval=jnp.pi)
    uav_pitch = jax.random.uniform(keys[6], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    init_dist = jax.random.uniform(keys[7], minval=2000.0, maxval=8000.0)
    init_azimuth = jax.random.uniform(keys[8], minval=-jnp.pi, maxval=jnp.pi)
    init_elevation = jax.random.uniform(keys[9], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    tar_pos = uav_pos + init_dist * jnp.array([
        jnp.cos(init_elevation) * jnp.cos(init_azimuth),
        jnp.cos(init_elevation) * jnp.sin(init_azimuth),
        jnp.sin(init_elevation)
    ])
    
    # 防止目標機鑽入地底
    tar_pos = tar_pos.at[2].set(jnp.maximum(500.0, tar_pos[2]))
    tar_yaw = jax.random.uniform(keys[10], minval=-jnp.pi, maxval=jnp.pi)
    tar_pitch = jax.random.uniform(keys[11], minval=-jnp.pi/6, maxval=jnp.pi/6)
    
    # 訓練時隨機抽樣所有 7 種目標運動模式 (包含 Weaving 蛇行等) 來強化訓練
    episode_target_motion = jax.random.randint(keys[15], shape=(), minval=0, maxval=7)
    tar_pitch = jnp.where(episode_target_motion == 0, 0.0, tar_pitch)
    
    dx = tar_pos[0] - uav_pos[0]
    dy = tar_pos[1] - uav_pos[1]
    dz = tar_pos[2] - uav_pos[2]
    prev_R = jnp.hypot(jnp.hypot(dx, dy), dz)
    
    yaw_los = jnp.arctan2(dy, dx)
    pitch_los = jnp.arctan2(dz, jnp.hypot(dx, dy) + 1e-6)
    yaw_err = wrap_pi(yaw_los - uav_yaw)
    pitch_err = wrap_pi(pitch_los - uav_pitch)
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
    
    # 隨機放大 1.1 ~ 1.5 倍，保留合理的繞路時間，避免強迫超出邊界
    random_offset = jax.random.uniform(keys[13], minval=1.1, maxval=1.5)
    t_impact = jnp.clip(t_min_intercept * random_offset, 0.0, params.max_time * 0.9)
    
    state = EnvState(
        uav_pos=uav_pos, uav_vel=uav_spd, uav_yaw=uav_yaw, uav_pitch=uav_pitch,
        uav_actual_pitch_accel=0.0, uav_actual_yaw_accel=0.0,
        uav_pitch_accel_rate=0.0, uav_yaw_accel_rate=0.0,
        tar_pos=tar_pos, tar_vel=tar_spd, tar_yaw=tar_yaw, tar_pitch=tar_pitch,
        prev_R=prev_R, sim_time=0.0, num_step=0,
        t_impact=t_impact, episode_target_motion=episode_target_motion,
        t0=0.0, prev_motion_type=0, nominal_closing_speed=nominal_closing_speed
    )
    
    obs = get_obs(keys[14], state, params)
    return obs, state
