from PredictionModule.TrackingDataset import TrackingLoader
import matplotlib.pyplot as plt
from PredictionModule.model import *
from PredictionModule.metrics import *


def visualize(data_csv, model_location):
    data = TrackingLoader(data_csv, 4, 3, max_tracks=100, step=1, visualize_mode=True)
    pasti = int(4 / 1) + 1
    future = int(3 / 1)
    model = MLP(pasti, future, num_neurons=64, hidden_layers=5)
    model.load_state_dict(torch.load(model_location))
    ades = []
    fdes = []
    length = 0
    for key, d in iter(data):
        fig, ax = plt.subplots(figsize=(10, 10))
        for object in d:
            length += 1
            past, future = object[:pasti, :-1], object[pasti:, :2]
            past_line, = ax.plot(past[:, 0], past[:, 1], marker='o')
            ax.plot(future[:, 0], future[:, 1], marker='v', color=past_line.get_color())

            gt = torch.tensor(past)
            gt = torch.unsqueeze(gt, 0).float().flatten(start_dim=1)
            output = model(gt).detach().numpy()
            output = output.squeeze(0).reshape(future[:, :2].shape)
            ax.plot(output[:, 0], output[:, 1], marker='x', color=past_line.get_color())

            ades.append(ADE_I(output, future))

        plt.savefig(f"Visuals/{key[0]}_{key[1]}.png")
        del fig
        plt.clf()

    # print(sum(ades) / length) #1.8217396998564481
    # print(sum(fdes) / length) #1.2850645390841946

    ades = np.array(ades)
    ades = np.mean(ades, axis=0)
    plt.clf()
    plt.plot(np.arange(0, len(ades) / 2, 0.5), ades, marker='o')
    plt.savefig("Average Displacement per timestep")
