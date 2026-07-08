from dataclasses import dataclass, field
from enum import Enum
from math import radians
import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MyEngagementMultiAgentEnv3D(MultiAgentEnv):
    metadata = {'render_mode': "human", 'render_fps': 60}

    dT                 = 0.016
    UAV_MAX_PITCH_ACCEL = 5000.0  # m/s^2
    UAV_MAX_YAW_ACCEL   = 5000.0  # m/s^2
    TAR_MAX_PITCH_ACCEL = 8000.0  # m/s^2
    TAR_MAX_YAW_ACCEL   = 8000.0  # m/s^2
    
    W_DIST             = 0.001
    MAX_STEPS          = 3000
    BASE_R             = 8000.0
    MAX_VEL            = 1500.0    # 速度正規化基準
    MAX_LOS_RATE       = 0.3       # LOS rate 正規化基準 (rad/s)

    @dataclass
    class ObjState3D:
        pos:     np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # x, y, z
        vel:     float = 0.0
        yaw:     float = 0.0  # 偏航角 (Heading/Yaw)
        pitch:   float = 0.0  # 俯仰角 (Pitch)
        actual_pitch_accel: float = 0.0
        actual_yaw_accel:   float = 0.0
        pitch_accel_rate:   float = 0.0
        yaw_accel_rate:     float = 0.0

        def __post_init__(self):
            if self.pos is None:
                self.pos = np.array([0.0, 0.0, 0.0])

    class TargetMotionType3D(Enum):
        Nothing           = 0
        PitchUp           = 1
        PitchDown         = 2
        YawLeft           = 3
        YawRight          = 4
        Weaving           = 5
        Spiral            = 6

    def __init__(self, config=None):
        super().__init__()

        self.agents = self.possible_agents = ['uav', 'target']

        # 🚀 3D 觀測特徵 (10維)：[R_norm, yaw_err, pitch_err, vel_norm, yaw_los, pitch_los, pitch, t_impact_norm, t_current_norm, time_deficit]
        self.observation_spaces = {
            'uav':    Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
            'target': Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
        }
        self.MAX_TIME = 30.0

        self.action_spaces = {
            'uav':    Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32), # [pitch_ratio, yaw_ratio]
            'target': Discrete(7),
        }

        self.target_move_state = {
            'motion_type':      self.TargetMotionType3D.Nothing,
            'prev_motion_type': self.TargetMotionType3D.Nothing,
            't0': 0.0,
            # 規避超參數 (隨優化變動或保持基準)
            'aw_yaw': self.TAR_MAX_YAW_ACCEL * 0.5,
            'aw_pitch': self.TAR_MAX_PITCH_ACCEL * 0.5,
            'Ww': 4.0,  # 擺動頻率
            'phiw': 0.0,
        }

        self.num_step = 0
        self.sim_time = 0.0
        self.prev_R   = None
        self.state    = {}
        
        config = config or {}
        self.force_target_action = bool(config.get("force_target_action", True))
        self.target_motion_mode = config.get("target_motion_mode", "random_episode")
        self.fixed_target_action = int(config.get("fixed_target_action", self.TargetMotionType3D.Nothing.value))
        self.target_spd_vray_rate = float(config.get("target_spd_vray_rate", 0.0))
        self.obs_noise_std = float(config.get("obs_noise_std", 0.0))
        self.use_2nd_order = bool(config.get("use_2nd_order", False)) # 新增二階開關
        self.target_impact_time = config.get("target_impact_time", None)
        if self.target_impact_time is not None:
            self.target_impact_time = float(self.target_impact_time)

        self.episode_target_motion = self.TargetMotionType3D.Nothing

    # ── 觀測輔助：從 A 的視角觀察 B (3D 幾何特徵) ──────────────────────────
    def _obs_helper(self, A: 'ObjState3D', B: 'ObjState3D') -> np.ndarray:
        dx = B.pos[0] - A.pos[0]
        dy = B.pos[1] - A.pos[1]
        dz = B.pos[2] - A.pos[2]
        R  = np.hypot(np.hypot(dx, dy), dz)
        
        # 3D 視線角
        yaw_los = np.arctan2(dy, dx)
        pitch_los = np.arctan2(dz, np.hypot(dx, dy) + 1e-6)
        
        # 朝向角誤差
        yaw_error = self._wrap_pi(yaw_los - A.yaw)
        pitch_error = self._wrap_pi(pitch_los - A.pitch)

        # 雙方速度在 3D 的向量表示
        ax_v = A.vel * np.cos(A.pitch) * np.cos(A.yaw)
        ay_v = A.vel * np.cos(A.pitch) * np.sin(A.yaw)
        az_v = A.vel * np.sin(A.pitch)
        
        bx_v = B.vel * np.cos(B.pitch) * np.cos(B.yaw)
        by_v = B.vel * np.cos(B.pitch) * np.sin(B.yaw)
        bz_v = B.vel * np.sin(B.pitch)

        dvx = bx_v - ax_v
        dvy = by_v - ay_v
        dvz = bz_v - az_v

        # 水平視線旋轉率
        r_xy = np.hypot(dx, dy)
        yaw_los_rate = (dx * dvy - dy * dvx) / (r_xy**2 + 1e-6)
        
        # 垂直視線旋轉率
        d_r_xy = (dx * dvx + dy * dvy) / (r_xy + 1e-6)
        pitch_los_rate = (r_xy * dvz - dz * d_r_xy) / (R**2 + 1e-6)

        obs = np.array([
            np.clip(R / self.BASE_R, 0.0, 1.0),                 # 相對距離歸一化
            yaw_error / np.pi,                                  # 偏航角誤差
            pitch_error / (np.pi / 2),                          # 俯仰角誤差
            np.clip(A.vel / self.MAX_VEL, -1.0, 1.0),           # 自身速度歸一化
            np.clip(yaw_los_rate / self.MAX_LOS_RATE, -1.0, 1.0), # 水平視線旋轉率
            np.clip(pitch_los_rate / self.MAX_LOS_RATE, -1.0, 1.0), # 垂直視線旋轉率
            A.pitch / (np.pi / 2),                              # 自身俯仰角歸一化
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_obs(self) -> dict:
        obs = {
            'uav':    self._obs_helper(self.state['uav'], self.state['target']),
            'target': self._obs_helper(self.state['target'], self.state['uav']),
        }
        
        # 僅對 7 維幾何觀測注入高斯噪聲，維持時間量測的精確性
        if self.obs_noise_std > 0.0:
            rng = self.np_random
            for aid in obs:
                noise = rng.normal(0.0, self.obs_noise_std, size=obs[aid].shape).astype(np.float32)
                obs[aid] = np.clip(obs[aid] + noise, -1.0, 1.0)
                
        # 計算正規化時間特徵 (範圍 [0.0, 1.0]) 和 time_deficit
        t_impact_norm = np.clip(self.t_impact / self.MAX_TIME, 0.0, 1.0)
        t_current_norm = np.clip(self.sim_time / self.MAX_TIME, 0.0, 1.0)
        
        t_ideal_remaining = self.t_impact - self.sim_time
        dx = self.state['target'].pos[0] - self.state['uav'].pos[0]
        dy = self.state['target'].pos[1] - self.state['uav'].pos[1]
        dz = self.state['target'].pos[2] - self.state['uav'].pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        t_straight_remaining = R / getattr(self, 'nominal_closing_speed', 100.0)
        time_deficit = np.clip((t_ideal_remaining - t_straight_remaining) / self.MAX_TIME, -1.0, 1.0)
        
        time_feats = np.array([t_impact_norm, t_current_norm, time_deficit], dtype=np.float32)
        
        # 拼接為 10 維觀測特徵
        for aid in obs:
            obs[aid] = np.concatenate([obs[aid], time_feats], axis=0)
            
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        
        do_random_start = options.get("random_start", True) if options else True

        self.num_step = 0
        self.sim_time = 0.0
        self.prev_R   = None
        
        self.episode_target_motion = self._select_episode_target_motion(rng)

        self.target_move_state.update({
            'motion_type': self.TargetMotionType3D.Nothing,
            'prev_motion_type': self.TargetMotionType3D.Nothing,
            't0': 0.0
        })

        if do_random_start or seed is not None:
            tar_spd = float(rng.uniform(600, 1500))
            uav_spd = tar_spd + float(rng.uniform(500, 1000))
            
            uav_pos = np.array([
                rng.uniform(-3000, 3000),
                rng.uniform(-3000, 3000),
                rng.uniform(3000, 8000)
            ], dtype=np.float32)
            
            init_dist = float(rng.uniform(2000, 8000))
            init_azimuth = float(rng.uniform(-np.pi, np.pi))
            init_elevation = float(rng.uniform(-np.pi/6, np.pi/6))
            
            tar_pos = uav_pos + init_dist * np.array([
                np.cos(init_elevation) * np.cos(init_azimuth),
                np.cos(init_elevation) * np.sin(init_azimuth),
                np.sin(init_elevation)
            ], dtype=np.float32)
            
            # 防止目標機鑽入地底
            tar_pos[2] = max(500.0, float(tar_pos[2]))
            
            self.state = {
                'uav': self.ObjState3D(
                    pos=uav_pos,
                    vel=uav_spd,
                    yaw=float(rng.uniform(-np.pi, np.pi)),
                    pitch=float(rng.uniform(-np.pi/6, np.pi/6)) # 初始俯仰角不宜過陡
                ),
                'target': self.ObjState3D(
                    pos=tar_pos,
                    vel=tar_spd,
                    yaw=float(rng.uniform(-np.pi, np.pi)),
                    pitch=float(rng.uniform(-np.pi/6, np.pi/6))
                ),
            }
        else:
            # 預設固定初始狀態
            self.state = {
                'uav': self.ObjState3D(
                    pos=np.array([-3000.0, -3000.0, 1500.0], dtype=np.float32), 
                    vel=1200.0, 
                    yaw=radians(45),
                    pitch=radians(0)
                ),
                'target': self.ObjState3D(
                    pos=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32), 
                    vel=500.0, 
                    yaw=radians(135),
                    pitch=radians(0)
                ),
            }

        # 如果為 Nothing 機動，強制將俯仰角 (pitch) 設為 0.0，實現水平直線飛行
        if self.episode_target_motion == self.TargetMotionType3D.Nothing:
            self.state['target'].pitch = 0.0

        # 必須在回傳 obs 前設定好初始 prev_R 供 step 時計算 Closing Rate
        dx = self.state['target'].pos[0] - self.state['uav'].pos[0]
        dy = self.state['target'].pos[1] - self.state['uav'].pos[1]
        dz = self.state['target'].pos[2] - self.state['uav'].pos[2]
        self.prev_R = np.hypot(np.hypot(dx, dy), dz)

        # 計算目標打擊時間估計
        dir_x = dx / (self.prev_R + 1e-6)
        dir_y = dy / (self.prev_R + 1e-6)
        dir_z = dz / (self.prev_R + 1e-6)
        
        tar_vel = self.state['target'].vel
        tar_pitch = self.state['target'].pitch
        tar_yaw = self.state['target'].yaw
        tar_vx = tar_vel * np.cos(tar_pitch) * np.cos(tar_yaw)
        tar_vy = tar_vel * np.cos(tar_pitch) * np.sin(tar_yaw)
        tar_vz = tar_vel * np.sin(tar_pitch)
        
        tar_flee_speed = tar_vx * dir_x + tar_vy * dir_y + tar_vz * dir_z
        self.nominal_closing_speed = max(100.0, self.state['uav'].vel - tar_flee_speed)
        
        t_min_intercept = self.prev_R / self.nominal_closing_speed
        
        # 隨機放大 1.1 ~ 1.5 倍
        random_offset = rng.uniform(1.1, 1.5)
        self.t_impact = np.clip(t_min_intercept * random_offset, 0.0, self.MAX_TIME * 0.9)

        return self._get_obs(), {aid: {} for aid in self.agents}

    def step(self, action_dict):
        self.num_step += 1
        self.sim_time  = self.num_step * self.dT

        M, T = self.state['uav'], self.state['target']

        # ── 擷取動作與控制 ────────────────────────────────────────────────
        # UAV 控制動作：action 為 [pitch_ratio, yaw_ratio]
        uav_act = action_dict.get('uav', np.array([0.0, 0.0]))
        cmd_pitch_accel = float(uav_act[0]) * self.UAV_MAX_PITCH_ACCEL
        cmd_yaw_accel   = float(uav_act[1]) * self.UAV_MAX_YAW_ACCEL

        # ── Autopilot 二階系統 ──────────────────────────────────────────────
        if self.use_2nd_order:
            tau = 0.707
            wn = 1.0 / tau
            zeta = 0.707
            
            # 俯仰加速度的二階差分更新
            M.pitch_accel_rate += self.dT * (wn**2 * (cmd_pitch_accel - M.actual_pitch_accel) - 2 * zeta * wn * M.pitch_accel_rate)
            M.actual_pitch_accel += self.dT * M.pitch_accel_rate
            
            # 偏航加速度的二階差分更新
            M.yaw_accel_rate += self.dT * (wn**2 * (cmd_yaw_accel - M.actual_yaw_accel) - 2 * zeta * wn * M.yaw_accel_rate)
            M.actual_yaw_accel += self.dT * M.yaw_accel_rate
            
            uav_pitch_accel = M.actual_pitch_accel
            uav_yaw_accel = M.actual_yaw_accel
        else:
            uav_pitch_accel = cmd_pitch_accel
            uav_yaw_accel = cmd_yaw_accel
            M.actual_pitch_accel = cmd_pitch_accel
            M.actual_yaw_accel = cmd_yaw_accel

        # Target 控制動作
        if self.force_target_action:
            T_a = self.episode_target_motion
        else:
            T_a_val = action_dict.get('target', self.TargetMotionType3D.Nothing.value)
            T_a = self.TargetMotionType3D(int(T_a_val))
            
        dx = T.pos[0] - M.pos[0]
        dy = T.pos[1] - M.pos[1]
        dz = T.pos[2] - M.pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        # 3D 視線角
        yaw_los = np.arctan2(dy, dx)
        pitch_los = np.arctan2(dz, np.hypot(dx, dy) + 1e-6)
        
        # 目標 3D 動作對應
        target_pitch_accel, target_yaw_accel = self.pick_target_move(
            T_a, self.sim_time, yaw_los, pitch_los, T.vel, T.yaw, T.pitch
        )

        # ── 3D 運動狀態物理更新 (UAV) ──────────────────────────────────────
        # 俯仰率與偏航率
        uav_dot_pitch = uav_pitch_accel / M.vel
        uav_cos_pitch_safe = max(0.01, np.cos(M.pitch))
        uav_dot_yaw = uav_yaw_accel / (M.vel * uav_cos_pitch_safe)
        
        # 更新角度並做物理裁剪
        M.pitch = np.clip(M.pitch + uav_dot_pitch * self.dT, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        M.yaw   = self._wrap_pi(M.yaw + uav_dot_yaw * self.dT)
        
        # 計算 3D 速度分量並更新位置
        M.pos[0] += M.vel * np.cos(M.pitch) * np.cos(M.yaw) * self.dT
        M.pos[1] += M.vel * np.cos(M.pitch) * np.sin(M.yaw) * self.dT
        M.pos[2] += M.vel * np.sin(M.pitch) * self.dT

        # ── 3D 運動狀態物理更新 (Target) ──────────────────────────────────
        target_dot_pitch = target_pitch_accel / T.vel
        target_cos_pitch_safe = max(0.01, np.cos(T.pitch))
        target_dot_yaw = target_yaw_accel / (T.vel * target_cos_pitch_safe)
        
        T.pitch = np.clip(T.pitch + target_dot_pitch * self.dT, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        T.yaw   = self._wrap_pi(T.yaw + target_dot_yaw * self.dT)

        if self.target_spd_vray_rate > 0.0:
            if self.np_random.uniform(0, 1) < self.target_spd_vray_rate:
                T.vel += self.np_random.uniform(
                    T.vel*-0.4, 
                    T.vel*0.4,
                    size=(1)
                )[0]
            # T.vel = np.clip(T.vel, self.TAR_MIN_SPD, self.TAR_MAX_SPD)
        
        T.pos[0] += T.vel * np.cos(T.pitch) * np.cos(T.yaw) * self.dT
        T.pos[1] += T.vel * np.cos(T.pitch) * np.sin(T.yaw) * self.dT
        T.pos[2] += T.vel * np.sin(T.pitch) * self.dT

        # ── 獎勵與距離計算 ────────────────────────────────────────────────
        new_R = np.hypot(np.hypot(T.pos[0] - M.pos[0], T.pos[1] - M.pos[1]), T.pos[2] - M.pos[2])
        closing_rate = (self.prev_R - new_R) / self.dT
        r_dist       = self.W_DIST * np.clip(closing_rate, -3000, 3000)
        r_step       = -0.005
        
        # 考慮加速度消耗懲罰
        uav_accel_norm = np.sqrt(uav_pitch_accel**2 + uav_yaw_accel**2)
        target_accel_norm = np.sqrt(target_pitch_accel**2 + target_yaw_accel**2)

        rewards = {
            'uav': float(r_dist + r_step - 0.01 * (uav_accel_norm / self.UAV_MAX_PITCH_ACCEL)),
            'target': float(-(r_dist + r_step) - np.abs(target_accel_norm) / self.TAR_MAX_PITCH_ACCEL)
        }

        self.prev_R = new_R

        # ── 終止條件判定 ──────────────────────────────────────────────────
        terminateds = {aid: False for aid in self.agents}
        truncateds  = {aid: False for aid in self.agents}
        terminateds['__all__'] = False
        truncateds['__all__']  = False
        
        hit = new_R < 30.0
        over = self.num_step >= self.MAX_STEPS
        out_of_bound = new_R > (self.BASE_R * 2)
        
        # 3D 新增終止條件：撞地判定 (高度 Z < 0)
        ground_crash = (M.pos[2] < 0.0) or (T.pos[2] < 0.0)

        # 隱式打擊時間控制 (Impact Time Control)：計算時間差
        time_err = abs(self.sim_time - self.t_impact)

        if hit:
            # 基礎擊中獎懲
            rewards['uav']    += 100.0
            rewards['target'] -= 100.0
            
            # 時間誤差懲罰：每偏離一秒扣除 15.0 分
            time_penalty = -4.0 * time_err
            rewards['uav'] += time_penalty
            
            # 高精度打擊獎勵 (誤差小於等於 0.2 秒)
            if time_err <= 0.2:
                rewards['uav'] += 100.0
                
            terminateds['__all__'] = True
            for aid in self.agents: terminateds[aid] = True
        elif ground_crash:
            # 懲罰墜地的載具
            if M.pos[2] < 0.0:
                rewards['uav'] -= 500.0
            if T.pos[2] < 0.0:
                rewards['target'] -= 500.0
            terminateds['__all__'] = True
            for aid in self.agents: terminateds[aid] = True
        elif over or out_of_bound:
            rewards['uav']    -= 20.0
            rewards['target'] += 20.0
            truncateds['__all__'] = True
            for aid in self.agents: truncateds[aid] = True

        infos = {
            'uav': {
                'pitch_accel': float(uav_pitch_accel),
                'yaw_accel': float(uav_yaw_accel),
                'ground_crash': bool(M.pos[2] < 0.0),
                't_impact': float(self.t_impact),
                'sim_time': float(self.sim_time),
                'time_err': float(time_err)
            },
            'target': {
                'pitch_accel': float(target_pitch_accel),
                'yaw_accel': float(target_yaw_accel),
                'ground_crash': bool(T.pos[2] < 0.0),
                'target_motion': int(T_a.value),
                'force_target_action': self.force_target_action,
            },
        }

        return self._get_obs(), rewards, terminateds, truncateds, infos

    def pick_target_move(self, T_a, sim_time, yaw_los, pitch_los, Vt, yaw_t, pitch_t, force_action=None) -> tuple:
        params = self.target_move_state
        params['motion_type'] = T_a

        if params['prev_motion_type'] != params['motion_type']:
            params['t0'] = sim_time
            params['prev_motion_type'] = T_a

        t = sim_time - params['t0']
        ap = 0.0  # pitch accel
        al = 0.0  # yaw accel
        
        T_a = force_action if force_action is not None else T_a
            
        if T_a == self.TargetMotionType3D.PitchUp:
            ap = self.TAR_MAX_PITCH_ACCEL
        elif T_a == self.TargetMotionType3D.PitchDown:
            ap = -self.TAR_MAX_PITCH_ACCEL
        elif T_a == self.TargetMotionType3D.YawLeft:
            al = self.TAR_MAX_YAW_ACCEL * 0.3
        elif T_a == self.TargetMotionType3D.YawRight:
            al = -self.TAR_MAX_YAW_ACCEL * 0.3
        elif T_a == self.TargetMotionType3D.Weaving:
            # 水平偏航方向正弦擺動
            al = params['aw_yaw'] * np.sin(params['Ww'] * t + params['phiw'])
        elif T_a == self.TargetMotionType3D.Spiral:
            # 3D 螺旋擺動規避 (俯仰與偏航呈現 90 度相位差的正弦)
            ap = params['aw_pitch'] * np.cos(params['Ww'] * t)
            al = params['aw_yaw'] * np.sin(params['Ww'] * t)

        return float(ap), float(al)
    
    def _select_episode_target_motion(self, rng):
        mode = self.target_motion_mode
        if mode == "none":
            return self.TargetMotionType3D.Nothing
        elif mode == "fixed":
            return self.TargetMotionType3D(int(self.fixed_target_action))
        elif mode == "random_episode":
            return rng.choice(list(self.TargetMotionType3D))
        elif mode == "random_episode_without_high_g":
            return rng.choice([
                self.TargetMotionType3D.Nothing,
                self.TargetMotionType3D.Weaving,
                self.TargetMotionType3D.Spiral,
            ])
        else:
            raise ValueError(f"Unknown target_motion_mode: {mode}")
        
    @staticmethod
    def _wrap_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi
