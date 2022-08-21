# from PredictionModule import visualize
# from PredictionModule.main import train
# import fire
#
# if __name__ == "__main__":
#     fire.Fire({
#         'visualize': visualize,
#         'train': train
#     })

import pyautogui
import time

for _ in range(int(550)):
    pyautogui.press("space")
    time.sleep(1)