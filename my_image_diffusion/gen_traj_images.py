from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def make_traj(start, goal, obstacle, rng, lim):
    n = 100
    step_size = 0.01
    step_noise = 0.01
    size = 35
    goal_potential = -5
    # simulate a dynamical system starting at the point (-1, -1) and moving towards (1, 1)
    # Let there be an obstacle at (0, 0), and it should avoid the obstacle.
    # we'll use a global potential field to simulate this
    # potential field using meshgrid
    x = np.linspace(0, lim, size)
    y = np.linspace(0, lim, size)
    X, Y = np.meshgrid(x, y)
    xy_field = np.stack([X, Y], axis=-1)
    obs_potential_field = -np.clip(1 / np.linalg.norm(xy_field - obstacle, axis=-1), 0, 10)
    goal_potential_field = goal_potential * np.linalg.norm(xy_field - goal, axis=-1)
    # combine the two potential fields
    potential_field = goal_potential_field + obs_potential_field
    # simulate the trajectory
    traj = [start]
    for i in range(n):
        current_pos = traj[-1]
        # use analytical gradient
        current_grad = get_gradient(current_pos, goal, goal_potential, obstacle)
        potential_next_pos = current_pos + step_size * current_grad
        next_grad = get_gradient(potential_next_pos, goal, goal_potential, obstacle)
        avg_grad = 0.5 * (current_grad + next_grad)
        next_pos = current_pos + step_size * avg_grad

        next_pos += rng.normal(0, step_noise, 2)
        traj.append(next_pos)

        if np.linalg.norm(next_pos - goal) < step_size:
            break
    traj_arr = np.array(traj)
    return traj_arr, potential_field


def get_gradient(current_pos, goal, goal_potential, obstacle):
    current_goal_grad = goal_potential * (current_pos - goal) / np.linalg.norm(current_pos - goal) ** 0.8
    current_obstacle_grad = 1.8 * (current_pos - obstacle) / np.linalg.norm(current_pos - obstacle) ** 1.3
    current_grad = current_goal_grad + current_obstacle_grad
    return current_grad


def main():
    start = np.array([0.1, 0.1])
    obstacle = np.array([1, 1])
    lim = 2

    root = Path("data/trajs")
    root.mkdir(exist_ok=True, parents=True)

    for i in range(250):
        rng = np.random.RandomState(i)
        goal = rng.rand(2) * 0.1 + 1.75
        traj, potential_field = make_traj(start, goal, obstacle, rng, lim)

        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], color='k')
        plt.scatter(obstacle[0], obstacle[1], color='red', s=100)
        plt.scatter(start[0], start[1], color='green')
        plt.scatter(goal[0], goal[1], color='blue')
        plt.imshow(np.flipud(potential_field), extent=(0, lim, 0, lim))
        plt.savefig(root / f"traj_{i}.png")


if __name__ == '__main__':
    main()
