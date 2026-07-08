import pygame
import numpy as np
import os
import sys
import collections

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️ 尚未安裝 onnxruntime，請執行 `pip install onnxruntime` 才能載入模型")

# ========================================================
# 1. 載入模型與配置
# ========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

TARGET_MODEL_PATH = os.path.join(current_dir, "jaxTraining", "models", "target_model_3d.onnx") 
UAV_MODEL_PATH = os.path.join(current_dir, "jaxTraining", "checkpoints", "checkpoint_20260706-1701", "uav_model_3d.onnx")
TARGET_MODEL_PATH = r"\\wsl.localhost\Ubuntu\home\willyliu\Code\Research\jaxTraining\checkpoints\checkpoint_20260707-1951\target_model_3d.onnx"
UAV_MODEL_PATH = r"\\wsl.localhost\Ubuntu\home\willyliu\Code\Research\jaxTraining\checkpoints\checkpoint_20260706-1701\uav_model_3d.onnx"
print("🔄 初始化 3D Target Evasion 模擬器...")

target_session = None
uav_session = None

if HAS_ONNX:
    try:
        target_session = ort.InferenceSession(TARGET_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"✅ 成功載入 ONNX Target 規避模型！: {TARGET_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ 載入 Target 權重模型失敗: {e}")
        
    try:
        uav_session = ort.InferenceSession(UAV_MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"✅ 成功載入 ONNX UAV 追擊模型！: {UAV_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ 載入 UAV 權重模型失敗: {e}")

# ========================================================
# 2. Numpy 環境模擬 (對齊 jax_env_target_3d.py 物理)
# ========================================================
class NumpyTargetEnv:
    def __init__(self):
        self.dt = 0.016
        self.uav_max_pitch_accel = 3000.0
        self.uav_max_yaw_accel = 3000.0
        self.tar_max_pitch_accel = 6000.0
        self.tar_max_yaw_accel = 6000.0
        self.base_r = 8000.0
        self.max_vel = 2500.0
        self.max_los_rate = 0.3
        self.max_time = 30.0
        
        self.history_len = 25
        self.history_interval = 12
        
        self.reset()
        
    def wrap_pi(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi
        
    def reset(self):
        self.tar_spd = np.random.uniform(300.0, 800.0)
        self.uav_spd = self.tar_spd + np.random.uniform(600.0, 1200.0)
        
        self.tar_pos = np.array([
            np.random.uniform(-3000.0, 3000.0),
            np.random.uniform(-3000.0, 3000.0),
            np.random.uniform(3000.0, 8000.0)
        ])
        
        init_dist = np.random.uniform(2000.0, 6000.0)
        init_azimuth = np.random.uniform(-np.pi, np.pi)
        init_elevation = np.random.uniform(-np.pi/6, np.pi/6)
        
        self.uav_pos = self.tar_pos + init_dist * np.array([
            np.cos(init_elevation) * np.cos(init_azimuth),
            np.cos(init_elevation) * np.sin(init_azimuth),
            np.sin(init_elevation)
        ])
        self.uav_pos[2] = max(1000.0, self.uav_pos[2])
        
        dx = self.tar_pos[0] - self.uav_pos[0]
        dy = self.tar_pos[1] - self.uav_pos[1]
        dz = self.tar_pos[2] - self.uav_pos[2]
        
        self.uav_yaw = np.arctan2(dy, dx) + np.random.uniform(-np.pi/4, np.pi/4)
        self.uav_pitch = np.arctan2(dz, np.hypot(dx, dy) + 1e-6) + np.random.uniform(-np.pi/6, np.pi/6)
        
        self.tar_yaw = self.wrap_pi(self.uav_yaw + np.pi) + np.random.uniform(-np.pi/4, np.pi/4)
        self.tar_pitch = np.random.uniform(-np.pi/6, np.pi/6)
        
        self.uav_actual_pitch_accel = 0.0
        self.uav_actual_yaw_accel = 0.0
        self.uav_pitch_accel_rate = 0.0
        self.uav_yaw_accel_rate = 0.0
        
        self.tar_actual_pitch_accel = 0.0
        self.tar_actual_yaw_accel = 0.0
        self.tar_pitch_accel_rate = 0.0
        self.tar_yaw_accel_rate = 0.0
        
        self.sim_time = 0.0
        self.num_step = 0
        
        R = np.hypot(np.hypot(dx, dy), dz)
        tar_yaw_los = np.arctan2(-dy, -dx)
        tar_pitch_los = np.arctan2(-dz, np.hypot(-dx, -dy) + 1e-6)
        tar_yaw_err = self.wrap_pi(tar_yaw_los - self.tar_yaw)
        tar_pitch_err = self.wrap_pi(tar_pitch_los - self.tar_pitch)
        init_feat = np.array([R, tar_yaw_err, tar_pitch_err])
        
        self.history_buffer = collections.deque(maxlen=25)
        for _ in range(25):
            self.history_buffer.append(init_feat)
            
        tar_vx = self.tar_spd * np.cos(self.tar_pitch) * np.cos(self.tar_yaw)
        tar_vy = self.tar_spd * np.cos(self.tar_pitch) * np.sin(self.tar_yaw)
        tar_vz = self.tar_spd * np.sin(self.tar_pitch)
        dir_x = dx / (R + 1e-6)
        dir_y = dy / (R + 1e-6)
        dir_z = dz / (R + 1e-6)
        tar_flee_speed = tar_vx * dir_x + tar_vy * dir_y + tar_vz * dir_z
        self.nominal_closing_speed = max(100.0, self.uav_spd - tar_flee_speed)
        t_min_intercept = R / self.nominal_closing_speed
        self.t_impact = np.clip(t_min_intercept * np.random.uniform(1.1, 1.5), 0.0, self.max_time * 0.9)
            
    def get_target_obs(self):
        dx = self.tar_pos[0] - self.uav_pos[0]
        dy = self.tar_pos[1] - self.uav_pos[1]
        dz = self.tar_pos[2] - self.uav_pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        yaw_los = np.arctan2(-dy, -dx)
        pitch_los = np.arctan2(-dz, np.hypot(-dx, -dy) + 1e-6)
        
        yaw_error = self.wrap_pi(yaw_los - self.tar_yaw)
        pitch_error = self.wrap_pi(pitch_los - self.tar_pitch)
        
        obs_current = np.array([
            np.clip(R / self.base_r, 0.0, 1.0),
            yaw_error / np.pi,
            pitch_error / (np.pi / 2),
            np.clip(self.tar_spd / self.max_vel, -1.0, 1.0),
            np.clip(self.uav_spd / self.max_vel, -1.0, 1.0),
            self.tar_pitch / (np.pi / 2)
        ], dtype=np.float32)
        obs_current = np.nan_to_num(obs_current, nan=0.0, posinf=1.0, neginf=-1.0)
        
        history_arr = np.array(self.history_buffer)
        R_hist = np.clip(history_arr[:, 0] / self.base_r, 0.0, 1.0)
        yaw_hist = history_arr[:, 1] / np.pi
        pitch_hist = history_arr[:, 2] / (np.pi / 2)
        obs_history = np.stack([R_hist, yaw_hist, pitch_hist], axis=-1).astype(np.float32)
        
        return obs_current, obs_history
        
    def get_uav_10dim_obs(self):
        dx = self.tar_pos[0] - self.uav_pos[0]
        dy = self.tar_pos[1] - self.uav_pos[1]
        dz = self.tar_pos[2] - self.uav_pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        yaw_los = np.arctan2(dy, dx)
        pitch_los = np.arctan2(dz, np.hypot(dx, dy) + 1e-6)
        yaw_error = self.wrap_pi(yaw_los - self.uav_yaw)
        pitch_error = self.wrap_pi(pitch_los - self.uav_pitch)
        
        ax_v = self.uav_spd * np.cos(self.uav_pitch) * np.cos(self.uav_yaw)
        ay_v = self.uav_spd * np.cos(self.uav_pitch) * np.sin(self.uav_yaw)
        az_v = self.uav_spd * np.sin(self.uav_pitch)
        
        bx_v = self.tar_spd * np.cos(self.tar_pitch) * np.cos(self.tar_yaw)
        by_v = self.tar_spd * np.cos(self.tar_pitch) * np.sin(self.tar_yaw)
        bz_v = self.tar_spd * np.sin(self.tar_pitch)
        
        dvx = bx_v - ax_v
        dvy = by_v - ay_v
        dvz = bz_v - az_v
        
        r_xy = np.hypot(dx, dy)
        yaw_los_rate = (dx * dvy - dy * dvx) / (r_xy**2 + 1e-6)
        d_r_xy = (dx * dvx + dy * dvy) / (r_xy + 1e-6)
        pitch_los_rate = (r_xy * dvz - dz * d_r_xy) / (R**2 + 1e-6)
        
        t_ideal_remaining = self.t_impact - self.sim_time
        t_straight_remaining = R / self.nominal_closing_speed
        time_deficit = np.clip((t_ideal_remaining - t_straight_remaining) / self.max_time, -1.0, 1.0)
        t_impact_norm = np.clip(self.t_impact / self.max_time, 0.0, 1.0)
        t_current_norm = np.clip(self.sim_time / self.max_time, 0.0, 1.0)
        
        obs = np.array([
            np.clip(R / self.base_r, 0.0, 1.0),
            yaw_error / np.pi,
            pitch_error / (np.pi / 2),
            np.clip(self.uav_spd / 1500.0, -1.0, 1.0),
            np.clip(yaw_los_rate / self.max_los_rate, -1.0, 1.0),
            np.clip(pitch_los_rate / self.max_los_rate, -1.0, 1.0),
            self.uav_pitch / (np.pi / 2),
            t_impact_norm,
            t_current_norm,
            time_deficit
        ], dtype=np.float32)
        return np.nan_to_num(obs)
        
    def get_png_pp_action(self):
        dx = self.tar_pos[0] - self.uav_pos[0]
        dy = self.tar_pos[1] - self.uav_pos[1]
        dz = self.tar_pos[2] - self.uav_pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        ax_v = self.uav_spd * np.cos(self.uav_pitch) * np.cos(self.uav_yaw)
        ay_v = self.uav_spd * np.cos(self.uav_pitch) * np.sin(self.uav_yaw)
        az_v = self.uav_spd * np.sin(self.uav_pitch)
        
        bx_v = self.tar_spd * np.cos(self.tar_pitch) * np.cos(self.tar_yaw)
        by_v = self.tar_spd * np.cos(self.tar_pitch) * np.sin(self.tar_yaw)
        bz_v = self.tar_spd * np.sin(self.tar_pitch)
        
        dv_x, dv_y, dv_z = bx_v - ax_v, by_v - ay_v, bz_v - az_v
        
        yaw_los = np.arctan2(dy, dx)
        pitch_los = np.arctan2(dz, np.hypot(dx, dy) + 1e-6)
        
        yaw_los_rate = (dx * dv_y - dy * dv_x) / (dx**2 + dy**2 + 1e-6)
        d_r_xy = (dx * dv_x + dy * dv_y) / (np.hypot(dx, dy) + 1e-6)
        pitch_los_rate = (np.hypot(dx, dy) * dv_z - dz * d_r_xy) / (R**2 + 1e-6)
        
        closing_vel = -(dx * dv_x + dy * dv_y + dz * dv_z) / (R + 1e-6)
        
        N = 3.0
        a_yaw_png = N * closing_vel * yaw_los_rate
        a_pitch_png = N * closing_vel * pitch_los_rate
        
        yaw_error = self.wrap_pi(yaw_los - self.uav_yaw)
        pitch_error = self.wrap_pi(pitch_los - self.uav_pitch)
        
        K_pp = 20.0
        a_yaw_pp = K_pp * self.uav_spd * yaw_error
        a_pitch_pp = K_pp * self.uav_spd * pitch_error
        
        alpha = np.clip((500.0 - R) / 500.0, 0.0, 1.0)
        
        uav_cmd_yaw_accel = (1.0 - alpha) * a_yaw_png + alpha * a_yaw_pp
        uav_cmd_pitch_accel = (1.0 - alpha) * a_pitch_png + alpha * a_pitch_pp
        
        return uav_cmd_pitch_accel, uav_cmd_yaw_accel
        
    def step(self, tar_act, uav_mode="PNG+PP"):
        tar_cmd_pitch_accel = np.clip(tar_act[0], -1.0, 1.0) * self.tar_max_pitch_accel
        tar_cmd_yaw_accel = np.clip(tar_act[1], -1.0, 1.0) * self.tar_max_yaw_accel
        
        if uav_mode == "ONNX" and uav_session is not None:
            uav_obs = self.get_uav_10dim_obs()
            uav_act = uav_session.run(None, {uav_session.get_inputs()[0].name: uav_obs.reshape(1, -1)})[0][0]
            # ONNX UAV is trained on max 5000, 1st order.
            uav_cmd_pitch_accel = np.clip(uav_act[0], -1.0, 1.0) * 5000.0
            uav_cmd_yaw_accel = np.clip(uav_act[1], -1.0, 1.0) * 5000.0
        else:
            uav_cmd_pitch_accel, uav_cmd_yaw_accel = self.get_png_pp_action()
            
        tau = 0.707
        wn = 1.0 / tau
        zeta = 0.707
        
        if uav_mode == "ONNX":
            # 1st order bypass for ONNX
            self.uav_actual_pitch_accel = uav_cmd_pitch_accel
            self.uav_actual_yaw_accel = uav_cmd_yaw_accel
            self.uav_pitch_accel_rate = 0.0
            self.uav_yaw_accel_rate = 0.0
        else:
            # 2nd order dynamic for PNG+PP
            self.uav_pitch_accel_rate += self.dt * (wn**2 * (uav_cmd_pitch_accel - self.uav_actual_pitch_accel) - 2 * zeta * wn * self.uav_pitch_accel_rate)
            self.uav_actual_pitch_accel += self.dt * self.uav_pitch_accel_rate
            self.uav_yaw_accel_rate += self.dt * (wn**2 * (uav_cmd_yaw_accel - self.uav_actual_yaw_accel) - 2 * zeta * wn * self.uav_yaw_accel_rate)
            self.uav_actual_yaw_accel += self.dt * self.uav_yaw_accel_rate
        
        self.tar_pitch_accel_rate += self.dt * (wn**2 * (tar_cmd_pitch_accel - self.tar_actual_pitch_accel) - 2 * zeta * wn * self.tar_pitch_accel_rate)
        self.tar_actual_pitch_accel += self.dt * self.tar_pitch_accel_rate
        self.tar_yaw_accel_rate += self.dt * (wn**2 * (tar_cmd_yaw_accel - self.tar_actual_yaw_accel) - 2 * zeta * wn * self.tar_yaw_accel_rate)
        self.tar_actual_yaw_accel += self.dt * self.tar_yaw_accel_rate
        
        self.uav_pitch = np.clip(self.uav_pitch + self.dt * self.uav_actual_pitch_accel / self.uav_spd, -np.pi/2 * 0.99, np.pi/2 * 0.99)
        self.uav_yaw = self.wrap_pi(self.uav_yaw + self.dt * self.uav_actual_yaw_accel / (self.uav_spd * np.cos(self.uav_pitch) + 1e-6))
        ax_v = self.uav_spd * np.cos(self.uav_pitch) * np.cos(self.uav_yaw)
        ay_v = self.uav_spd * np.cos(self.uav_pitch) * np.sin(self.uav_yaw)
        az_v = self.uav_spd * np.sin(self.uav_pitch)
        self.uav_pos += self.dt * np.array([ax_v, ay_v, az_v])
        
        self.tar_pitch = np.clip(self.tar_pitch + self.dt * self.tar_actual_pitch_accel / self.tar_spd, -np.pi/2 * 0.99, np.pi/2 * 0.99)
        self.tar_yaw = self.wrap_pi(self.tar_yaw + self.dt * self.tar_actual_yaw_accel / (self.tar_spd * np.cos(self.tar_pitch) + 1e-6))
        bx_v = self.tar_spd * np.cos(self.tar_pitch) * np.cos(self.tar_yaw)
        by_v = self.tar_spd * np.cos(self.tar_pitch) * np.sin(self.tar_yaw)
        bz_v = self.tar_spd * np.sin(self.tar_pitch)
        self.tar_pos += self.dt * np.array([bx_v, by_v, bz_v])
        
        self.sim_time += self.dt
        self.num_step += 1
        
        dx = self.tar_pos[0] - self.uav_pos[0]
        dy = self.tar_pos[1] - self.uav_pos[1]
        dz = self.tar_pos[2] - self.uav_pos[2]
        R = np.hypot(np.hypot(dx, dy), dz)
        
        if self.num_step % self.history_interval == 0:
            tar_yaw_los = np.arctan2(-dy, -dx)
            tar_pitch_los = np.arctan2(-dz, np.hypot(-dx, -dy) + 1e-6)
            tar_yaw_err = self.wrap_pi(tar_yaw_los - self.tar_yaw)
            tar_pitch_err = self.wrap_pi(tar_pitch_los - self.tar_pitch)
            feat = np.array([R, tar_yaw_err, tar_pitch_err])
            self.history_buffer.append(feat)
            
        done = False
        status = ""
        if R < 30.0:
            done = True
            status = "UAV Hit"
        elif self.tar_pos[2] < 0:
            done = True
            status = "Target Crashed"
        elif self.uav_pos[2] < 0:
            done = True
            status = "UAV Crashed"
        elif R > self.base_r * 4.0:
            done = True
            status = "Target Escaped (Out of Bounds)"
        elif self.sim_time >= self.max_time:
            done = True
            status = "Target Survived (Timeout)"
            
        return R, done, status

# ========================================================
# 3. Pygame 與 3D 渲染視角設定
# ========================================================
pygame.init()
pygame.font.init()

GAME_WIDTH = 800
PANEL_WIDTH = 360
WIDTH, HEIGHT = GAME_WIDTH + PANEL_WIDTH, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Target Evasion Sim (ONNX)")
clock = pygame.time.Clock()

cam_yaw = np.radians(-45)
cam_pitch = np.radians(25)
SCALE = 0.06
CX, CY = GAME_WIDTH // 2, HEIGHT // 2 + 100
dragging_rot = False
dragging_pan = False
last_mouse_pos = (0, 0)

def project_3d_to_2d(pos):
    x, y, z = pos
    cos_y, sin_y = np.cos(cam_yaw), np.sin(cam_yaw)
    x1 = x * cos_y - y * sin_y
    y1 = x * sin_y + y * cos_y
    z1 = z
    cos_p, sin_p = np.cos(cam_pitch), np.sin(cam_pitch)
    x2 = x1
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p
    return int(CX + x2 * SCALE), int(CY - z2 * SCALE)

def draw_3d_agent(surf, color, pos, heading_yaw, heading_pitch, radius=12):
    sx, sy = project_3d_to_2d(pos)
    vx = radius * 1.5 * np.cos(heading_pitch) * np.cos(heading_yaw)
    vy = radius * 1.5 * np.cos(heading_pitch) * np.sin(heading_yaw)
    vz = radius * 1.5 * np.sin(heading_pitch)
    h_pos = [pos[0] + vx, pos[1] + vy, pos[2] + vz]
    hx, hy = project_3d_to_2d(h_pos)
    pygame.draw.circle(surf, color, (sx, sy), radius)
    pygame.draw.line(surf, (255, 255, 255), (sx, sy), (hx, hy), 2)
    shx, shy = project_3d_to_2d([pos[0], pos[1], 0.0])
    pygame.draw.line(surf, (90, 100, 115), (sx, sy), (shx, shy), 1)
    pygame.draw.ellipse(surf, (40, 48, 62), (shx - 8, shy - 4, 16, 8))

# ========================================================
# 4. 主迴圈
# ========================================================
if __name__ == "__main__":
    font = pygame.font.SysFont("consolas", 14)
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    
    env = NumpyTargetEnv()
    uav_mode = "ONNX" if uav_session is not None else "PNG+PP"
    running = True
    paused = False
    R = 99999.0
    
    uav_traj = []
    tar_traj = []
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env.reset()
                    uav_traj.clear()
                    tar_traj.clear()
                elif event.key == pygame.K_u:
                    if uav_mode == "PNG+PP" and uav_session is not None:
                        uav_mode = "ONNX"
                    else:
                        uav_mode = "PNG+PP"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging_rot = True
                    last_mouse_pos = event.pos
                elif event.button == 3:
                    dragging_pan = True
                    last_mouse_pos = event.pos
                elif event.button == 4:
                    SCALE *= 1.1
                elif event.button == 5:
                    SCALE /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_rot = False
                elif event.button == 3:
                    dragging_pan = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging_rot:
                    dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                    cam_yaw -= dx * 0.01
                    cam_pitch = np.clip(cam_pitch - dy * 0.01, -np.pi/2, np.pi/2)
                    last_mouse_pos = event.pos
                elif dragging_pan:
                    dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                    CX += dx
                    CY += dy
                    last_mouse_pos = event.pos
    
        if not paused:
            if target_session is not None:
                obs_curr, obs_hist = env.get_target_obs()
                tar_act = target_session.run(None, {
                    target_session.get_inputs()[0].name: obs_curr.reshape(1, -1),
                    target_session.get_inputs()[1].name: obs_hist.reshape(1, 25, 3)
                })[0][0]
            else:
                tar_act = np.array([0.0, 0.0], dtype=np.float32)
    
            R, done, status = env.step(tar_act, uav_mode)
            
            uav_traj.append(np.copy(env.uav_pos))
            tar_traj.append(np.copy(env.tar_pos))
            if len(uav_traj) > 500:
                uav_traj.pop(0)
                tar_traj.pop(0)
                
            if done:
                print(f"End condition: {status}")
                env.reset()
                uav_traj.clear()
                tar_traj.clear()
    
        screen.fill((20, 24, 33))
        
        # 繪製 3D 地面參考網格 (Ground Grid at Z=0)
        grid_color = (55, 65, 80)
        grid_extent = 6000
        grid_step = 1000
        
        # 繪製平行於 Y 軸的網格線
        for gx in range(-grid_extent, grid_extent + 1, grid_step):
            pts = []
            for gy in range(-grid_extent, grid_extent + 1, grid_step * 2):
                pts.append(project_3d_to_2d([gx, gy, 0.0]))
            pygame.draw.lines(screen, grid_color, False, pts, 1)
            
        # 繪製平行於 X 軸的網格線
        for gy in range(-grid_extent, grid_extent + 1, grid_step):
            pts = []
            for gx in range(-grid_extent, grid_extent + 1, grid_step * 2):
                pts.append(project_3d_to_2d([gx, gy, 0.0]))
            pygame.draw.lines(screen, grid_color, False, pts, 1)

        # 繪製 Z 軸 (垂直中軸線)
        origin_shadow = project_3d_to_2d([0.0, 0.0, 0.0])
        origin_top = project_3d_to_2d([0.0, 0.0, 3000.0])
        pygame.draw.line(screen, (75, 85, 100), origin_shadow, origin_top, 1)
        
        # 畫軌跡
        if len(uav_traj) > 1:
            pts = [project_3d_to_2d(p) for p in uav_traj]
            pygame.draw.lines(screen, (200, 50, 50), False, pts, 2)
        if len(tar_traj) > 1:
            pts = [project_3d_to_2d(p) for p in tar_traj]
            pygame.draw.lines(screen, (50, 150, 250), False, pts, 2)
            
        draw_3d_agent(screen, (255, 80, 80), env.uav_pos, env.uav_yaw, env.uav_pitch, 8)
        draw_3d_agent(screen, (100, 200, 255), env.tar_pos, env.tar_yaw, env.tar_pitch, 8)
        
        pygame.draw.rect(screen, (30, 35, 45), (GAME_WIDTH, 0, PANEL_WIDTH, HEIGHT))
        y_offset = 20
        texts = [
            ("--- Controls ---", (200, 200, 200)),
            ("U: Toggle UAV Mode", (255, 255, 100)),
            ("Space: Pause/Resume", (200, 200, 200)),
            ("R: Reset", (200, 200, 200)),
            ("Mouse L/R/Wheel: Camera", (200, 200, 200)),
            ("", (200, 200, 200)),
            (f"Time: {env.sim_time:.2f} s", (200, 200, 200)),
            (f"Dist: {R:.1f} m", (255, 100, 100) if R < 500 else (100, 255, 100)),
            ("", (200, 200, 200)),
            (f"[UAV Mode: {uav_mode}]", (255, 150, 150)),
            (f"UAV Vel:   {env.uav_spd:.1f} m/s", (255, 100, 100)),
            (f"UAV Z:     {env.uav_pos[2]:.1f} m", (255, 100, 100)),
            ("", (200, 200, 200)),
            ("[Target Mode: ONNX (Jax)]", (150, 200, 255)),
            (f"Tar Vel:   {env.tar_spd:.1f} m/s", (100, 200, 255)),
            (f"Tar Z:     {env.tar_pos[2]:.1f} m", (100, 200, 255)),
            (f"Tar Pitch: {np.degrees(env.tar_pitch):.1f} deg", (100, 200, 255)),
            (f"Tar Yaw:   {np.degrees(env.tar_yaw):.1f} deg", (100, 200, 255)),
        ]
        
        for txt, col in texts:
            s = font.render(txt, True, col)
            screen.blit(s, (GAME_WIDTH + 20, y_offset))
            y_offset += 25
            
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
