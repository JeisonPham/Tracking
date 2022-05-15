from PredictionModule import visualize
from PredictionModule.main import train
import fire

if __name__ == "__main__":
    fire.Fire({
        'visualize': visualize,
        'train': train
    })
