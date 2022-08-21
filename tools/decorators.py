import matplotlib.pyplot as plt


class SavePlots(object):
    def __init__(self, save_location:str):
        self.index = 0
        self.location = save_location

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            plt.figure(figsize=(10, 10))
            fig = plt.figure()

            result = func(*args, **kwargs)
            if fig.get_axes():
                plt.savefig(f"{self.location}/{self.index:03d}.png")
                plt.show()
                self.index += 1
            else:
                plt.close(fig)


            return result
        return wrapper