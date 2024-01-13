from typing import Dict, Text
import pandas as pd
import numpy as np
import pickle
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "vehicles_count": 0,
            "vehicles_density": 1,
            "simulation_frequency": 25,  # [Hz]
            "policy_frequency": 25,  # [Hz]
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "other_vehicles_type": "highway_env.vehicle.new_behavior.IDMVehicle",
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": 0,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [470, 80, 100, 550]  # Before, converging, merge, after
        self.tot_len = ends[0] + ends[1] + ends[3]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [-StraightLane.DEFAULT_WIDTH, 0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [s, s], [n, c]]
        line_type_merge = [[c, s], [s, s], [n, s]]
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(10, 0),
                                                     speed=-5)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for id in range(self.config["vehicles_count"]):
            vehicle = other_vehicles_type.create_random_merge(self.road, spacing=1 / self.config["vehicles_density"], speed=30)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

            new_record = np.array(
                [[vehicle.position[0], vehicle.velocity[0], vehicle.action['acceleration'], 0, self.time]])
            vehicle.past_record.append(new_record)
            self.vehicle = vehicle

        self.vehicle = ego_vehicle

    def _simulate(self, action) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"], self.time)
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        if self.time > 650:
            terminated = True

        self._spawn_vehicle()
        self._spawn_merge_vehicle()
        self._clear_vehicles()

        return obs, reward, terminated, truncated, info
    
    def _spawn_merge_vehicle(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for lane_id in ['j']:
            if int(self.time*10) % 50 == 0:
                vehicle = other_vehicles_type.create_random_merge(self.road, spacing=1 / self.config["vehicles_density"],
                                                                  add=True, lane_from=lane_id)
                if vehicle:
                    vehicle.randomize_behavior()
                    vehicle.merge = True
                    self.road.vehicles.append(vehicle)

                    new_record = np.array(
                        [[vehicle.position[0], vehicle.velocity[0], vehicle.action['acceleration'], 0, self.time]])
                    vehicle.past_record.append(new_record)

    def _spawn_vehicle(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for lane_id in ['a']:
            if int(self.time*10) % 2 == 0:
                vehicle = other_vehicles_type.create_random_merge(self.road, spacing=1 / self.config["vehicles_density"],
                                                                  add=True, lane_from=lane_id)
                if vehicle:
                    vehicle.randomize_behavior()
                    self.road.vehicles.append(vehicle)

                    new_record = np.array(
                        [[vehicle.position[0], vehicle.velocity[0], vehicle.action['acceleration'], 0, self.time]])
                    vehicle.past_record.append(new_record)

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= self.tot_len - 4 * vehicle.LENGTH

        for vehicle in self.road.vehicles:
            if is_leaving(vehicle) and vehicle.lane_index[0] != 'j':
                self.road.deleted_vehicles.append(vehicle)

        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not is_leaving(vehicle)]

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
5