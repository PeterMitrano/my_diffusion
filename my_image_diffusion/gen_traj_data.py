from pathlib import Path

import numpy as np


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

    n_samples = 250
    trajs_dataset = []
    for i in range(n_samples):
        rng = np.random.RandomState(i)
        goal = rng.rand(2) * 0.1 + 1.75
        traj, potential_field = make_traj(start, goal, obstacle, rng, lim)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], color='k')
        plt.scatter(obstacle[0], obstacle[1], color='red', s=100)
        plt.scatter(start[0], start[1], color='green')
        plt.scatter(goal[0], goal[1], color='blue')
        plt.imshow(np.flipud(potential_field), extent=(0, lim, 0, lim))
        # turn of axes and remove padding
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(root / f"traj_{i}.png")

        # downsample to a fixed length
        # time = 50
        # i = np.linspace(0, len(traj) - 1, time)
        # l = np.floor(i).astype(int)
        # l = np.clip(l, 0, len(traj) - 2)
        # alpha = (i - l)[:, None]
        # traj_interp = (1 - alpha) * traj[l] + alpha * traj[l + 1]
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(traj[:, 0], traj[:, 1], color='k')
        # plt.show()
        # trajs_dataset.append(traj_interp)
    # np.save(root / f"trajs.npy", trajs_dataset)


if __name__ == '__main__':
    main()
