from RadarDataset import RadarDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from sklearn.cluster import DBSCAN
from Trajectory import Trajectory
from scipy.spatial.distance import directed_hausdorff
import pickle


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


def create_mask(fileLocation, carFile, cloudFile):
    dataset = RadarDataset(fileLocation, carFile, cloudFile, only_y=True, ego_only=False, flip_aug=False, t_spacing=0.25,
                           is_train=False)
    print(len(dataset))
    trajectory_set = []
    for i, y in enumerate(iter(dataset)):
        idxes = np.array(np.where(y == 1))[1:, :].T
        plt.plot(idxes[:, 0], idxes[:, 1])
        # temp = np.zeros((16, 2))
        # temp[:len(idxes), :] = idxes
        # idxes = temp
        if len(trajectory_set) == 0:
            trajectory_set.append(Trajectory(idxes))
        else:
            distances = [x.distance(idxes) for x in trajectory_set]
            if np.all(np.isnan(distances)):
                trajectory_set.append(Trajectory(idxes))
            else:
                ii = np.nanargmin(distances)

                if distances[ii] < 5:
                    trajectory_set[ii].update(idxes)
                else:
                    trajectory_set.append(Trajectory(idxes))

    # for i, y in enumerate(iter(dataset)):
    #     idxes = np.array(np.where(y == 1))[1:, :].T
    #     distances = [x.distance(idxes) for x in trajectory_set]
    #     if np.all(np.isnan(distances)):
    #         raise "error"
    plt.show()
    trajectory_set_index = {}
    for i, traj in enumerate(trajectory_set):
        avg = traj.mean
        plt.plot(avg[:, 0], avg[:, 1])
        trajectory_set_index[i] = traj
    plt.show()
    print(len(trajectory_set))

    with open("Trajectory_set.pkl", "wb") as file:
        pickle.dump(trajectory_set_index, file)
    #
    # traj_count = len(trajectory_set)
    # D = np.zeros((traj_count, traj_count))
    # for i in range(traj_count):
    #     for j in range(i + 1, traj_count):
    #         distance = L2(trajectory_set[i], trajectory_set[j])
    #         D[i, j] = distance
    #         D[j, i] = distance
    #
    # mdl = DBSCAN(eps=400, min_samples=10)
    # cluster_lst = mdl.fit_predict(D)
    # plot_cluster(trajectory_set, cluster_lst)


if __name__ == "__main__":
    create_mask("../Data", "downtown_SD_10thru_50count_with_cad_id.csv", "downtown_SD_10_7.ply")
    # t = np.linspace(1, 3, 1000)
    # t = fresnel(t)
    # plt.plot(t[1], -t[0])
    # plt.show()
