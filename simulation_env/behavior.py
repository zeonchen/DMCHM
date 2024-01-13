from typing import Tuple, Union
import numpy as np
from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
import torch
import random
import pickle


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 5 # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 beta=0.9):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        if random.random() <= 0.9:
            beta = 0.1 

        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.last_time_wanted = 1.5
        self.beta = beta

        self.lc_model = torch.load('best_model.pth')
        self.scaler = pickle.load(open('scaler.pkl', 'rb'))
        self.past_rv = None
        self.merge = False
        
        self.reset_dmchm()

    def reset_dmchm(self):
        self.z_prev_sv, self.z_prev_fv = self.lc_model.z_0_sv.unsqueeze(0), self.lc_model.z_0_fv.unsqueeze(0)
        self.l_prev_sv, self.l_prev_fv = self.lc_model.l_0_sv.unsqueeze(0), self.lc_model.l_0_fv.unsqueeze(0)
        self.sv1_prev, self.sv2_prev, self.sv3_prev, self.sv4_prev = self.lc_model.p_0_sv1.unsqueeze(
            0), self.lc_model.p_0_sv2.unsqueeze(0), \
                                                                     self.lc_model.p_0_sv3.unsqueeze(
                                                                         0), self.lc_model.p_0_sv4.unsqueeze(0)
        self.fv1_prev, self.fv2_prev, self.fv3_prev, self.fv4_prev = self.lc_model.p_0_fv1.unsqueeze(
            0), self.lc_model.p_0_fv2.unsqueeze(0), \
                                                                     self.lc_model.p_0_fv3.unsqueeze(
                                                                         0), self.lc_model.p_0_fv4.unsqueeze(0)
        self.game_inter = False

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        if self.velocity[0] < 2 and not self.merge:
            action['steering'] = self.steering_control(self.target_lane_index, abortion=True)
            action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
            self.target_lane_index = self.lane_index

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        
        if ('b', 'c', 3) in self.road.network.side_lanes(self.lane_index) or self.merge:
            self.COMFORT_ACC_MAX = 10
            self.last_time_wanted = -10
        else:
            self.COMFORT_ACC_MAX = 3

        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)

        if ('b', 'c', 3) in self.road.network.side_lanes(self.lane_index):
            merge_front_vehicle, _ = self.road.neighbour_vehicles(self, ('b', 'c', 3))

            if merge_front_vehicle and merge_front_vehicle.position[0] != 650 and abs(self.position[0] - merge_front_vehicle.position[0]) > 6:
                merge_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                           front_vehicle=merge_front_vehicle,
                                                           rear_vehicle=None)
                action['acceleration'] = min(action['acceleration'], merge_idm_acceleration)

        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index and not self.merge:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)

        if self.lane_index != self.target_lane_index and self.merge:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            
            action['acceleration'] = target_idm_acceleration

        if self.game_inter:
            game_front_vehicle, _ = self.road.neighbour_vehicles(self, self.game_inter)
            game_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                      front_vehicle=front_vehicle,
                                                      rear_vehicle=None)
            action['acceleration'] = min(action['acceleration'], game_idm_acceleration)
            self.game_inter = None

        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        if self.velocity[0] < 0.:
            action['acceleration'] = np.clip(action['acceleration'], 0, self.ACC_MAX)

        if self.merge and self.position[0] > 650:
            self.past_record = self.past_record[-1:]
            self.past_prob = self.past_prob[-1:]
            self.lc_count = 0
            self.merge = False

        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float, time):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt, time)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = 30
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(
            max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)), self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)

        return acceleration
 
    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = 0.8 * self.last_time_wanted + 0.2 * (1 + (1 - self.beta) * 2)
        tau = np.clip(tau, 1, 2)
        self.last_time_wanted = tau
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star 

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        if self.position[0] < 50:
            return

        # If a lane change is already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        if self.merge and self.position[0] > 550:
            cur_fv, _ = self.road.neighbour_vehicles(self, self.lane_index)
            tar_fv, tar_rv = self.road.neighbour_vehicles(self, ('b', 'c', 2))

            new_following_pred_a = self.acceleration(ego_vehicle=tar_rv, front_vehicle=self)

            if new_following_pred_a < -10:
                return 

            if tar_rv and self.position[0] - tar_rv.position[0] < 5:
                return

            # Do I have a planned route for a specific lane which is safe for me to access?
            self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=tar_fv)
            if self_pred_a < -10:
                return 

            if self.velocity[0] < 1:
                self.target_lane_index = ('b', 'c', 2)
                return
            
            self.target_lane_index = ('b', 'c', 2)

            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # Motivation?
        self.past_len = 20
        cur_fv, _ = self.road.neighbour_vehicles(self, self.lane_index)

        if str(cur_fv).split(' ')[0] != 'IDMVehicle':
            cur_fv = None

        if cur_fv is None:
            return

        sv_state = np.concatenate(self.past_record[-self.past_len:]).mean(axis=0)
        cur_fv_state = np.concatenate(cur_fv.past_record[-self.past_len:]).mean(axis=0)
        cur_fv_v = cur_fv_state[1]
        sv_state = sv_state[1]
        speed_disadvantage = (cur_fv_v - sv_state) / sv_state
        if speed_disadvantage > 0.5:
            return

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            if lane_index == ('b', 'c', 3):
                continue
            # Speed Advantage?
            tar_fv, tar_rv = self.road.neighbour_vehicles(self, lane_index)

            if str(tar_fv).split(' ')[0] != 'IDMVehicle':
                tar_fv = None
            if str(tar_rv).split(' ')[0] != 'IDMVehicle':
                tar_rv = None

            if tar_fv is None:
                continue

            tar_fv_state = np.concatenate(tar_fv.past_record[-self.past_len:]).mean(axis=0)
            tar_fv_v = tar_fv_state[1]
            speed_advantage = (tar_fv_v - cur_fv_v) / tar_fv_v

            if speed_advantage < 0.1:
                continue

            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Only change lane when the vehicle is moving
            if np.abs(self.speed) < 0.1:
                continue

            if self.dmchm(lane_index):
                self.target_lane_index = lane_index

    def dmchm(self, lane_index):
        cur_fv, _ = self.road.neighbour_vehicles(self, self.lane_index)
        tar_fv, tar_rv = self.road.neighbour_vehicles(self, lane_index)

        new_following_pred_a = self.acceleration(ego_vehicle=tar_rv, front_vehicle=self)

        if tar_rv is None or cur_fv is None or tar_fv is None:
            return self.mobil(lane_index)

        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=tar_fv)
        if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        if int(str(tar_rv).split(':')[0].split('#')[1]) != self.past_rv:
            self.reset_dmchm()
            self.past_rv = int(str(tar_rv).split(':')[0].split('#')[1])
            self.past_prob = []

        cur_fv_state = np.concatenate(cur_fv.past_record[-self.past_len:]).mean(axis=0)
        tar_fv_state = np.concatenate(tar_fv.past_record[-self.past_len:]).mean(axis=0)
        tar_rv_state = np.concatenate(tar_rv.past_record[-self.past_len:]).mean(axis=0)
        sv_state = np.concatenate(self.past_record[-self.past_len:]).mean(axis=0)

        # calculate 14 features
        feat = np.array([sv_state[1], cur_fv_state[1],
                         tar_fv_state[1], tar_rv_state[1],
                         tar_fv_state[2], tar_rv_state[2],
                         sv_state[2], cur_fv_state[2],
                         sv_state[0] - tar_rv_state[0], sv_state[0] - cur_fv_state[0],
                         sv_state[0] - tar_fv_state[0],
                         sv_state[1] - cur_fv_state[1],
                         sv_state[1] - tar_fv_state[1],
                         sv_state[1] - tar_rv_state[1]]).reshape(1, -1)
        feat = torch.from_numpy(self.scaler.transform(feat)).float()

        # update
        self.z_prev_sv, self.z_prev_fv, self.l_prev_sv, self.l_prev_fv, \
        self.sv1_prev, self.sv2_prev, self.sv3_prev, self.sv4_prev, \
        self.fv1_prev, self.fv2_prev, self.fv3_prev, self.fv4_prev, \
        emission_probs_t_sv, emission_probs_t_fv = \
        self.lc_model.stepwise_val(self.z_prev_sv, self.z_prev_fv, torch.tensor([self.beta]).unsqueeze(0), torch.tensor([self.beta]).unsqueeze(0),
                                   self.l_prev_sv, self.l_prev_fv,
                                   self.sv1_prev, self.sv2_prev, self.sv3_prev, self.sv4_prev,
                                   self.fv1_prev, self.fv2_prev, self.fv3_prev, self.fv4_prev, cur_obs=feat)
        if emission_probs_t_fv < 0.5:
            tar_rv.game_inter = self.lane_index
        else:
            tar_rv.game_inter = None
        self.past_prob.append(emission_probs_t_sv.item())
            
        if emission_probs_t_sv >= 0.5:
            self.lc_count += 1
            self.past_prob = []
            return True
        else:
            return False

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class LinearVehicle(IDMVehicle):

    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.5

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] -
                                                                        self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle)))

    def acceleration_features(self, ego_vehicle: ControlledVehicle,
                              front_vehicle: Vehicle = None,
                              rear_vehicle: Vehicle = None) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.speed),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2)])
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Target speed dynamics
        phi0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])
        # Front speed control
        phi1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]
        ])
        # Front position control
        phi2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 1, -self.TIME_WANTED, 0],
            [0, 0, 0, 0]
        ])
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        phi0 = np.array([
            [0, 0],
            [0, -1]
        ])
        phi1 = np.array([
            [0, 0],
            [-1, 0]
        ])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]
