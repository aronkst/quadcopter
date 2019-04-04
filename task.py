import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        angular_v0 = self.calcule(self.sim.angular_v[0], 0, 10)
        angular_v1 = self.calcule(self.sim.angular_v[1], 0, 10)
        angular_v2 = self.calcule(self.sim.angular_v[2], 0, 10)
        angular_v = angular_v0 + angular_v1 + angular_v2

        pose0 = self.calcule(self.sim.pose[0], 0, 100)
        pose1 = self.calcule(self.sim.pose[1], 0, 100)
        pose2 = self.calcule(self.sim.pose[2], 100, 100)
        pose = pose0 + pose1 + (pose2 * 3)

        reward = angular_v + pose
        return reward

    def calcule(self, current, goal, interval):
        if current == goal:
            return interval * 2

        if current < (goal - interval) or current > (goal + interval):
            return 0

        if goal == 0:
            if current > goal:
                return interval - current
            else:
                return current + interval
        else:
            if current > goal:
                return (interval + goal) - current
            else:
                return current

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def new_step(self, rotor_speeds):
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, self.sim.pose, self.sim.angular_v

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
