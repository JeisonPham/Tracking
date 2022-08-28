import open3d.visualization.gui as gui


def Open3DErrorProtect(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(str(e))
            gui.Application.instance.quit()

    return wrapper
