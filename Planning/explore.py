from RadarDataset import RadarDataset
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from Trajectory import Trajectory
from scipy.spatial.distance import directed_hausdorff
import pickle
from tqdm import tqdm


def hausdorff(u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def L2(u, v):
    d = []
    N = min(len(u), len(v))
    for i in range(N):
        d.append(np.linalg.norm(u[i, :] - v[i, :]))
    return max(d)


color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                  'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])


def plot_cluster(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    cluster_count = np.max(cluster_lst) + 1

    for traj, cluster in zip(traj_lst, cluster_lst):

        if cluster == -1:
            # Means it is a noisy trajectory, paint it black
            plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')

        else:
            plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    plt.show()





def create_trajectory_set(dataset):
    direction = {
        "less_than_5": {
            0: [],
            1: [],
            2: [],
            3: []
        },
        "greater_than_5": {
            0: [],
            1: [],
            2: [],
            3: []
        },
    }

    for i, (_, y, (speed, dire)) in enumerate(iter(dataset)):
        if speed <= 5:
            speed = "less_than_5"
        else:
            speed = "greater_than_5"

        trajectory_set = direction[speed][dire]
        y = y.detach().cpu().numpy()

        if len(trajectory_set) == 0:
            trajectory_set.append(Trajectory(y))
        else:
            distances = [x.distance(y) for x in trajectory_set]
            if np.all(np.isnan(distances)):
                trajectory_set.append(Trajectory(y))
            else:
                ii = np.nanargmin(distances)
                if distances[ii] < 5:
                    trajectory_set[ii].update(y)
                else:
                    trajectory_set.append(Trajectory(y))
        direction[speed][dire] = trajectory_set

    for k, v in direction.items():
        for key, value in v.items():
            trajectory_set = {}
            for i, traj in enumerate(value):
                mean = traj.mean
                plt.plot(mean[:, 0], mean[:, 1])
                trajectory_set[i] = traj
            direction[k][key] = trajectory_set
            plt.title(f"{k}_{key}")
            plt.xlim([0, 255])
            plt.ylim([0, 255])
            plt.show()

    with open("Trajectory_set.pkl", "wb") as file:
        pickle.dump(direction, file)


def create_mask(dataset, out_name):
    import json

    print(len(dataset))
    masks = np.zeros((len(dataset.local_ts),
                      dataset.nx,
                      dataset.ny))
    for ix, (_, ys, _) in enumerate(tqdm(dataset)):
        update = (ys.sum(0) > 0).numpy()
        masks[update] = 1.0
        # symmetry
        masks[np.flip(update, 2)] = 1.0

    print('saving', out_name)
    with open(out_name, 'w') as writer:
        json.dump(masks.tolist(), writer)


def viz_masks(out_name, imname='masks.jpg'):
    import json
    import matplotlib as mpl

    """Visualize the masks.
    """
    with open(out_name, 'r') as reader:
        data = json.load(reader)

    fig = plt.figure(figsize=(16 * 4, 1 * 4))
    gs = mpl.gridspec.GridSpec(1, 16)

    for maski, mask in enumerate(data):
        plt.subplot(gs[0, maski])
        plt.imshow(np.array(mask).T, origin='lower', vmin=0, vmax=1)
        plt.title(f'N Occupied cells: {np.array(mask).sum()}')

    plt.tight_layout()
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)


def viz_trajectory_sets(traj_file):
    with open(traj_file, 'rb') as file:
        trajectory_set = pickle.load(file)

    for key_speed, value_speed in trajectory_set.items():
        for key_dir, value_dir in value_speed.items():
            plt.title(f"{key_speed} {key_dir}")
            plt.xlim([0, 255])
            plt.ylim([0, 255])
            for traj in value_dir.values():
                plt.plot(traj.mean[:, 0], traj.mean[:, 1])
            plt.show()


if __name__ == "__main__":
    # create_mask("../Data", "downtown_SD_10thru_50count_with_cad_id.csv", "downtown_SD_10_7.ply")
    viz_trajectory_sets("Trajectory_set.pkl")
    # t = np.linspace(1, 3, 1000)
    # t = fresnel(t)
    # plt.plot(t[1], -t[0])
    # plt.show()
