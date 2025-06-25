import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_env.SnekEnv import SnekEnv

KEY_TO_ACTION = {
    81: 0,  # flèche gauche
    83: 1,  # flèche droite
    84: 2,  # flèche bas
    82: 3,  # flèche haut
}


def main():
    env = SnekEnv(render_mode="human")
    obs, info = env.reset()
    done = False

    print("Control the Snake wit hkeyboard arrows")
    print("Press 'q' to exit")

    while not done:
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        action = KEY_TO_ACTION.get(key, None)
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
