#-*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import math
import random

from numpy import *
from matplotlib import *

with open('./05_TrajectoryPlanning/map/map_coord_proto2.pkl', 'rb') as f:
    map_coord = pickle.load(f)

map_in = map_coord['Lane_inner']
map_center = map_coord['Lane_center']
map_out = map_coord['Lane_outer']
wp_in = map_coord['waypoint_inner']
wp_out = map_coord['waypoint_outer']

# initialize
V_MAX = 2      # maximum velocity [m/s]
ACC_MAX = 2 # maximum acceleration [m/ss]
K_MAX = 5     # maximum curvature [1/m]
V_MIN = 0.1

TARGET_SPEED = 1 # target speed [m/s]
LANE_WIDTH = 0.39  # lane width [m]

COL_CHECK = 0.2 # collision check distance [m]

MIN_T = 1 # minimum terminal time [s]
MAX_T = 2 # maximum terminal time [s]
DT_T = 0.5 # dt for terminal time [s] : MIN_T 에서 MAX_T 로 어떤 dt 로 늘려갈지를 나타냄
DT = 0.1 # timestep for update

# cost weights
K_J_lat = 0.0 # weight for lateral jerk
K_J_lon = 0.0 # weight for longitudinal jerk
K_T = 0.0 # weight for terminal time
K_D = 0.0 # weight for consistency
K_V = 0.0 # weight for getting to target speed
K_LAT = 0.0 # weight for lateral direction
K_LON = 0.0 # weight for longitudinal direction

SIM_STEP = 300 # Total simulation time (1/10 Scale)
SHOW_MAP = False # 지도 plot 여부
SHOW_ANIMATION = True # plot 으로 결과 보여줄지 말지

# Vehicle parameters - plot 을 위한 파라미터
LENGTH = 0.39  # [m]
WIDTH = 0.19  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.22  # [m]

# lateral planning 시 terminal position condition 후보  (양 차선 중앙)
DF_SET = np.array([LANE_WIDTH/2, -LANE_WIDTH/2])
# Longitudinal velocity condition 후보 (현재 속도 += 0.5)
SF_D_SET = np.array([-0.5, 0.0, 0.5])
# Map 시각화
def plot_map_with_sparse_waypoints(map_in, map_center, map_out, wp_in, wp_out):
    """
    차선 경로와 웨이포인트를 시각화합니다.
    웨이포인트는 10개 간격으로만 플롯되며, 인덱스 번호도 함께 표시됩니다.
    """

    def to_xy_array(path):
        path = np.array(path)
        return path.T if path.shape[1] == 2 else path

    def plot_path(path, label, style='-'):
        xy = to_xy_array(path)
        plt.plot(xy[0], xy[1], style, label=label)

    def plot_sparse_waypoints(wps, label, color='r'):
        wps = np.array(wps)
        sparse_idx = np.arange(0, len(wps), 10)
        sparse_wps = wps[sparse_idx]
        xy = sparse_wps.T

        plt.scatter(xy[0], xy[1], c=color, s=30, label=label, zorder=5)

        for i, (x, y) in zip(sparse_idx, sparse_wps):
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.figure(figsize=(10, 8))
    
    # Plot lanes
    plot_path(map_in, 'Lane Inner', style='--')
    plot_path(map_center, 'Lane Center', style='-')
    plot_path(map_out, 'Lane Outer', style='--')

    # Plot 10-step waypoints
    plot_sparse_waypoints(wp_in, 'Waypoint Inner', color='blue')
    plot_sparse_waypoints(wp_out, 'Waypoint Outer', color='green')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Lane Map with Every 10th Waypoint Indexed')
    plt.show()
    
# Get next waypoint considering vehicle position
def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)
    next_idx = np.mod((closest_wp + 1),len(mapx))

    map_vec = [mapx[next_idx] - mapx[closest_wp], mapy[next_idx] - mapy[closest_wp]]
    ego_vec = [x - mapx[closest_wp], y - mapy[closest_wp]]

    direction  = np.sign(np.dot(map_vec, ego_vec))

    if direction >= 0:
        next_wp = next_idx
    else:
        next_wp = closest_wp
    return next_wp

# Get nearest waypoint from the vehicle
def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closeset_wp = 0

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp

# Calculate Euclidean distance between 2 points
def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x)**2 + (y - _y)**2)

# Transfer to frenet coordinate
def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    prev_wp = next_wp -1

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
    proj_x = proj_norm*n_x
    proj_y = proj_norm*n_y

    #-------- get frenet d
    frenet_d = get_dist(x_x,x_y,proj_x,proj_y)

    ego_vec = [x-mapx[prev_wp], y-mapy[prev_wp], 0];
    map_vec = [n_x, n_y, 0];
    d_cross = np.cross(ego_vec,map_vec)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d;

    #-------- get frenet s
    frenet_s = 0;
    for i in range(prev_wp):
        frenet_s = frenet_s + get_dist(mapx[i],mapy[i],mapx[i+1],mapy[i+1]);

    frenet_s = frenet_s + get_dist(0,0,proj_x,proj_y);

    return frenet_s, frenet_d

# Transfer to cartesian coordinate
def get_cartesian(s, d, mapx, mapy, maps):
    prev_wp = 0

    s = np.mod(s, maps[-2])

    while(s > maps[prev_wp+1]) and (prev_wp < len(maps)-2):
        prev_wp = prev_wp + 1

    next_wp = np.mod(prev_wp+1,len(mapx))

    dx = (mapx[next_wp]-mapx[prev_wp])
    dy = (mapy[next_wp]-mapy[prev_wp])

    heading = np.arctan2(dy, dx) # [rad]

    # the x,y,s along the segment
    seg_s = s - maps[prev_wp];

    seg_x = mapx[prev_wp] + seg_s*np.cos(heading);
    seg_y = mapy[prev_wp] + seg_s*np.sin(heading);

    perp_heading = heading + 90 * np.pi/180;
    x = seg_x + d*np.cos(perp_heading);
    y = seg_y + d*np.sin(perp_heading);

    return x, y, heading

class QuinticPolynomial:

    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # calculate coefficient of quintic polynomial
        # used for lateral trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T** 4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2,
                      vf - self.a1 - 2*self.a2*T,
                      af - 2*self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    # calculate position info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5 * t ** 5
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return j

class QuarticPolynomial:

    def __init__(self, xi, vi, ai, vf, af, T):
        # calculate coefficient of quartic polynomial
        # used for longitudinal trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[3*T**2, 4*T**3],
                             [6*T, 12*T**2]])
        b = np.array([vf - self.a1 - 2*self.a2*T,
                             af - 2*self.a2])

        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    # calculate position info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t
        return j

class FrenetPath:

    def __init__(self):
        # time
        self.t = []

        # lateral traj in Frenet frame
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # longitudinal traj in Frenet frame
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        # cost
        self.c_lat = 0.0
        self.c_lon = 0.0
        self.c_tot = 0.0

        # combined traj in global frame
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.kappa = []
class Obstacle:
    def __init__(self,int_lat, int_lon, s, d, s_d, d_d, mapx, mapy, maps):
            self.freq_LC = int_lat
            self.intention_lat = 0
            self.intention_lon = int_lon
            self.s = s
            self.d = d
            self.s_d = s_d
            self.d_d = d_d
            _x, _y, _ = get_cartesian(self.s,self.d, mapx, mapy, maps)
            self.x = _x
            self.y = _y
            self.psi = np.arcsin(np.clip(d/LANE_WIDTH*2,-1.0,1.0))
        
class Obstacles:
    def __init__(self, obs_init, mapx, mapy, maps):
        self.num_obs = len(obs_init)
        self.obstacles = []
        self.mapx = mapx
        self.mapy = mapy
        self.maps = maps
        for ob in obs_init:
            obs = Obstacle(ob[3], 0.0, ob[0], ob[1], ob[2], 0.0, self.mapx, self.mapy, self.maps)
            self.obstacles.append(obs)
    def update_obs(self, host_s, host_d, dt=DT):
        for obs in (self.obstacles):
            lane_change_prob = 0.05
            if LANE_WIDTH/2 - obs.d < 0.01: 
                if obs.intention_lat != -1.0:
                    r = random.random()
                    if r < lane_change_prob:
                        obs.intention_lat = -1.0
                    else:
                        obs.intention_lat = 0.0
            elif -LANE_WIDTH/2 - obs.d > -0.01:
                if obs.intention_lat != 1.0:
                    r = random.random()
                    if r < lane_change_prob:
                        obs.intention_lat = 1.0
                    else:
                        obs.intention_lat = 0.0
            else:
                obs.intention_lat = (1.0 if random.random() < 0.995 else -1.0) * obs.intention_lat
            
            # 주변 object 상호작용
            other_s_list = [other.s for other in self.obstacles if other is not obs]
            other_d_list = [other.d for other in self.obstacles if other is not obs]
            other_s_list.append(host_s)
            other_d_list.append(host_d)
            for i in range(len(other_s_list)):
                # Longitudinal intention
                if obs.s < 0.4:
                    obs.intention_lon = 1
                    break
                elif other_s_list[i] - obs.s < 0.5 and other_s_list[i] - obs.s > 0.0 and np.sign(obs.d) == np.sign(other_d_list[i]):
                    obs.intention_lon = -1
                    break
                elif obs.s_d < 0.5:
                    obs.intention_lon = 1
                    break
                elif obs.s_d > 0.7:
                    obs.intention_lon = -1
                else:
                    obs.intention_lon = 0
                # Lateral intention
                if other_s_list[i] - obs.s < 1.0 and other_s_list[i] - obs.s > 0.0:
                    obs.intention_lat = -np.sign(other_d_list[i])
                    break
                elif other_s_list[i] - obs.s <= 0.0 and other_s_list[i] - obs.s > -0.2:
                    obs.intention_lat = -np.sign(other_d_list[i])
                    break
            
            # Speed update (Intention)
            if obs.intention_lon < 0:
                obs.s_d -= 0.05
            elif obs.intention_lon > 0:
                obs.s_d += 0.05
                
            if obs.intention_lat < 0:
                # obs.d_d = -LANE_WIDTH/(0.5/dt)
                obs.d_d = -LANE_WIDTH/(0.5/dt) if obs.d > -LANE_WIDTH/2 - 0.01 else 0
            elif obs.intention_lat > 0:
                # obs.d_d = LANE_WIDTH/(0.5/dt)
                obs.d_d = LANE_WIDTH/(0.5/dt) if obs.d < LANE_WIDTH/2 + 0.01 else 0
            else:
                obs.d_d = 0.0
            # Position update (CV)
            obs.s = obs.s + obs.s_d*dt
            obs.s = np.mod(obs.s, self.maps[-2])
            obs.d = obs.d + obs.d_d*dt
            _x, _y, _ = get_cartesian(obs.s, obs.d, self.mapx, self.mapy, self.maps)
            obs.x = _x
            obs.y = _y

def calc_frenet_paths(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d):
    frenet_paths = []

    # generate path to each offset goal
    for sf_d_diff in SF_D_SET:
        for df in DF_SET:
            # Lateral motion planning
            for T in np.arange(MIN_T, MAX_T+DT_T, DT_T):
                fp = FrenetPath()
                lat_traj = QuinticPolynomial(di, di_d, di_dd, df, df_d, df_dd, T)

                fp.t = [t for t in np.arange(0.0, T, DT)]
                fp.d = [lat_traj.calc_pos(t) for t in fp.t]
                fp.d_d = [lat_traj.calc_vel(t) for t in fp.t]
                fp.d_dd = [lat_traj.calc_acc(t) for t in fp.t]
                fp.d_ddd = [lat_traj.calc_jerk(t) for t in fp.t]

                # Longitudinal motion planning (velocity keeping)
                tfp = deepcopy(fp)
                lon_traj = QuarticPolynomial(si, si_d, si_dd, sf_d+sf_d_diff, sf_dd, T)

                tfp.s = [lon_traj.calc_pos(t) for t in fp.t]
                tfp.s_d = [lon_traj.calc_vel(t) for t in fp.t]
                tfp.s_dd = [lon_traj.calc_acc(t) for t in fp.t]
                tfp.s_ddd = [lon_traj.calc_jerk(t) for t in fp.t]

                # 경로 늘려주기 (In case T < MAX_T)
                for _t in np.arange(T, MAX_T, DT):
                    tfp.t.append(_t)
                    tfp.d.append(tfp.d[-1])
                    _s = tfp.s[-1] + tfp.s_d[-1] * DT
                    tfp.s.append(_s)

                    tfp.s_d.append(tfp.s_d[-1])
                    tfp.s_dd.append(tfp.s_dd[-1])
                    tfp.s_ddd.append(tfp.s_ddd[-1])

                    tfp.d_d.append(tfp.d_d[-1])
                    tfp.d_dd.append(tfp.d_dd[-1])
                    tfp.d_ddd.append(tfp.d_ddd[-1])

                J_lat = sum(np.power(tfp.d_ddd, 2))/(MAX_T/DT)  # lateral jerk
                J_lon = sum(np.power(tfp.s_ddd, 2))/(MAX_T/DT)  # longitudinal jerk

                # cost for consistency
                d_diff = (tfp.d[-1] - opt_d) ** 2
                # cost for target speed
                v_diff = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                # lateral cost
                tfp.c_lat = K_J_lat * J_lat + K_T * T + K_D * d_diff
                # longitudinal cost
                tfp.c_lon = K_J_lon * J_lon + K_T * T + K_V * v_diff
                # total cost
                tfp.c_tot = K_LAT * tfp.c_lat + K_LON * tfp.c_lon

                frenet_paths.append(tfp)
    return frenet_paths

def calc_global_paths(fplist, mapx, mapy, maps):

    # transform trajectory from Frenet to Global
    for fp in fplist:
        for i in range(len(fp.s)):
            _s = fp.s[i]
            _d = fp.d[i]
            _x, _y, _ = get_cartesian(_s, _d, mapx, mapy, maps)
            fp.x.append(_x)
            fp.y.append(_y)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            fp.kappa.append(yaw_diff / fp.ds[i])

    return fplist

def get_predicted_trajectory(ob, T_pred=MAX_T, DT_pred=DT): # CV model for now(frenet coordinate)
    # t_pred = [t for t in np.arange(0, T_pred, DT_pred)]
    s_pred = [ob.s+ob.s_d*t for t in np.arange(0, T_pred, DT_pred)]
    d_pred = [ob.d for t in np.arange(0, T_pred, DT_pred)]
    return s_pred, d_pred
        
        

def collision_check(fp, obs, mapx, mapy, maps):
    # Code    
    # collision = Trajectoy 를 따라 충돌여부 판단(COL_CHECK 거리 이내에 존재 여부)

    if collision:
        return True
    return False


def check_path(fplist, obs, mapx, mapy, maps):
    # Code
    # 경로의 Constraints 체크 (속도, 가속도, 곡률, 충돌여부)
    ok_ind = []
    # 유효한 경로만 Append
    ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d):
    fplist = calc_frenet_paths(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, opt_d)
    fplist = calc_global_paths(fplist, mapx, mapy, maps)

    fplist = check_path(fplist, obs, mapx, mapy, maps)
    # find minimum cost path
    min_cost = float("inf")
    opt_traj = None
    opt_ind = 0
    for fp in fplist:
        if min_cost >= fp.c_tot:
            min_cost = fp.c_tot
            opt_traj = fp
            _opt_ind = opt_ind
        opt_ind += 1
    # No solution error
    try:
        _opt_ind
    except NameError:
        print(" No solution ! ")

    return fplist, _opt_ind



def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def OptimalTraj_main():
    # map waypoints
    if SHOW_MAP == True:
        plot_map_with_sparse_waypoints(map_in, map_center, map_out, wp_in, wp_out)
    mapx = map_center[:,0]
    mapy = map_center[:,1]
    maplen = 0.0
    for i in range(len(mapx)):
        maplen += np.sqrt((mapx[i]-mapx[i-1])**2+(mapy[i]-mapy[i-1])**2)

    # get maps
    maps = np.zeros(mapx.shape)
    for i in range(len(mapx)):
        x = mapx[i]
        y = mapy[i]
        sd = get_frenet(x, y, mapx, mapy)
        maps[i] = sd[0]

    # Dynamic obstacles ( Frenet coordinate position )
    obs_init = np.array([[1.5, LANE_WIDTH/2, 0.6, 0.1]
                    ,[5.0, -LANE_WIDTH/2, 0.4, 0.1]
                    ,[7.0, LANE_WIDTH/2, 0.6, 0.3]
                    ,[8.5, -LANE_WIDTH/2, 0.4, 0.2]
                   ])
   
    obs = Obstacles(obs_init, mapx, mapy, maps)
    obs_global = []
    for i, obs_ in enumerate(obs.obstacles):
        _s = obs_.s
        _d = obs_.d
        xy = get_cartesian(_s, _d, mapx, mapy, maps)
        obs_global.append(xy[:-1])

    # 자챠량 관련 initial condition
    x = -LANE_WIDTH/2
    y = 0
    yaw = 90 * np.pi/180
    v = 0.5
    a = 0

    s, d = get_frenet(x, y, mapx, mapy);
    x, y, yaw_road = get_cartesian(s, d, mapx, mapy, maps)
    yawi = yaw - yaw_road

    # s 방향 초기조건
    si = s
    si_d = v*np.cos(yawi)
    si_dd = a*np.cos(yawi)
    sf_d = TARGET_SPEED
    sf_dd = 0

    # d 방향 초기조건
    di = d
    di_d = v*np.sin(yawi)
    di_dd = a*np.sin(yawi)
    df_d = 0
    df_dd = 0

    opt_d = di

    # 시뮬레이션 수행 (SIM_STEP 만큼)
    plt.figure(figsize=(14,10))
    for step in range(SIM_STEP):
        # optimal planning 수행 (output : valid path & optimal path index)
        path, opt_ind = frenet_optimal_planning(si, si_d, si_dd,
                                                sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d)
        # Vehicle position update(control)
        '''
        다음 시뮬레이션 step 에서 사용할 initial condition update.
        본 파트에서는 planning 만 수행하고 control 은 따로 수행하지 않으므로,
        optimal trajectory 중 현재 위치에서 한 cycle 뒤 index 를 다음 step 의 초기초건으로 사용.
        '''
        si = path[opt_ind].s[1]
        si_d = path[opt_ind].s_d[1]
        si_dd = path[opt_ind].s_dd[1]
        di = path[opt_ind].d[1]
        di_d = path[opt_ind].d_d[1]
        di_dd = path[opt_ind].d_dd[1]
        
        # Obstacle update
        obs.update_obs(path[opt_ind].s[1], path[opt_ind].d[1])
        
        # Lap count for simulation
        sim_lap = si//maplen

        # consistency cost를 위해 update
        opt_d = path[opt_ind].d[-1]
        if SHOW_ANIMATION:  # pragma: no cover
            plt.clf()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            # cartesian frame
            plt.subplot(1,2,1)
            # plot map
            plt.plot(map_center[:,0], map_center[:,1], 'k', linewidth=2)
            plt.plot(map_in[:,0], map_in[:,1], 'k', linewidth=2)
            plt.plot(map_out[:,0], map_out[:,1], 'k', linewidth=2)
            plt.plot(wp_in[:,0], wp_in[:,1], color='slategray', linewidth=2, alpha=0.5)
            plt.plot(wp_out[:,0], wp_out[:,1], color='slategray', linewidth=2, alpha=0.5)
            # plot obstacle
            for ob in obs.obstacles:
                plt.plot(ob.x, ob.y, marker = "s", markersize = 15, color="crimson", alpha=0.6)
            # plot path
            for i in range(len(path)):
                    plt.plot(path[i].x, path[i].y, "-", color="crimson", linewidth=1.5, alpha=0.6)
            plt.plot(path[opt_ind].x, path[opt_ind].y, "o-", color="dodgerblue", linewidth=3)
            # plot car
            plot_car(path[opt_ind].x[0], path[opt_ind].y[0], path[opt_ind].yaw[0], steer=0)
            # plot setting
            plt.axis('equal')
            plt.title("Real world")
            plt.grid(True)
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            
            # frenet frame
            plt.subplot(1,2,2)
            # plot map
            plt.plot([0, 0], [path[opt_ind].s[0]-1.0, path[opt_ind].s[0]+3.0], 'k', linewidth=2)
            plt.plot([-LANE_WIDTH, -LANE_WIDTH], [path[opt_ind].s[0]-1.0, path[opt_ind].s[0]+3.0], 'k', linewidth=2)
            plt.plot([LANE_WIDTH, LANE_WIDTH], [path[opt_ind].s[0]-1.0, path[opt_ind].s[0]+3.0], 'k', linewidth=2)
            plt.plot([LANE_WIDTH/2, LANE_WIDTH/2], [path[opt_ind].s[0]-1.0, path[opt_ind].s[0]+3.0], color='slategray', linewidth=2, alpha=0.5)
            plt.plot([-LANE_WIDTH/2, -LANE_WIDTH/2], [path[opt_ind].s[0]-1.0, path[opt_ind].s[0]+3.0], color='slategray', linewidth=2, alpha=0.5)
            # plot obstacle
            for ob in obs.obstacles:
                plt.plot(-ob.d,ob.s+sim_lap*maplen, marker = "s", markersize = 25, color="crimson", alpha=0.6)
                plt.plot(-ob.d,ob.s+(sim_lap+1)*maplen, marker = "s", markersize = 25, color="crimson", alpha=0.6)
            # plot path
            for i in range(len(path)):
                plt.plot(-np.array(path[i].d), np.array(path[i].s),"-", color="crimson", linewidth=1.5, alpha=0.6)
            plt.plot(-np.array(path[opt_ind].d), np.array(path[opt_ind].s), "o-", color="dodgerblue", linewidth=3)
            # plot car
            plot_car(-np.array(path[opt_ind].d[0]), np.array(path[opt_ind].s[0]), np.pi/2, steer=0)
            # plot setting
            plt.axis([-1.0, 1.0, path[opt_ind].s[0]-0.5, path[opt_ind].s[0]+2.5])
            plt.title("Frenet coordinate")
            plt.grid(True)
            plt.xlabel("d [m]")
            plt.ylabel("s [m]")
            
            

            plt.suptitle("[Simulation] v : " + str(si_d)[0:4] + " m/s")
            
            plt.pause(0.01)
            # input("Press enter to continue...")


if __name__ == "__main__":
    OptimalTraj_main()
