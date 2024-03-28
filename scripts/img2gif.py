import os
import argparse

import imageio


def compose_gif(img_path, gif_name, fps):
    image_na = sorted(os.listdir(img_path))
    gif_images = []
    for path in image_na:
        if path.endswith(".png"):
            gif_images.append(imageio.imread(img_path + path))

    imageio.mimsave(gif_name, gif_images, duration=fps)


def main():
    parsers = argparse.ArgumentParser(description="Compose gif from images")
    parsers.add_argument("--img_path", type=str, default="renders/waymo/waymo_debug/")
    parsers.add_argument("--gif_name", type=str, default="renders/waymo/waymo_debug/waymo_debug.gif")
    parsers.add_argument("--fps", type=int, default=10)
    img_path, gif_name, fps = parsers.parse_args().img_path, parsers.parse_args().gif_name, parsers.parse_args().fps
    compose_gif(img_path, gif_name, fps)


if __name__ == "__main__":
    main()
