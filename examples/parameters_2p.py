import os.path

import numpy as np

import utils_video
import utils_video.generators
import utils2p


# @profile
def main():
    #experiments = [
    #    "/mnt/data/FA/200612_ABO_GC7c_tdTom/Fly1/001_coronal",
    #    "/mnt/data/FA/200612_ABO_GC7c_tdTom/Fly1/002_coronal",
    #    "/mnt/data/FA/200615_ABO_GC7c_tdTom/Fly1/001_coronal",
    #    "/mnt/data/FA/200615_ABO_GC7c_tdTom/Fly2/002_coronal",
    #    "/mnt/data/FA/200615_ABO_GC7c_tdTom/Fly3/001_coronal",
    #    "/mnt/data/FA/200617_ABO_GC7c_tdTom/Fly1/002_coronal",
    #    "/mnt/data/FA/200623_ABO_GC7c_tdTom/Fly1/001_coronal",
    #    "/mnt/data/FA/200626_ABO_GC7c_tdTom/Fly1/002_coronal",
    #]
    #experiments = [
    #    "/mnt/data/FA/191129_ABO/Fly1/001_coronal",
    #    "/mnt/data/FA/191129_ABO/Fly2/001_coronal",
    #    "/mnt/data/FA/200514_G23xM4/Fly1/002_coronal/",
    #]

    experiments = [
        "/mnt/data/FA/200703_G23xM4/Fly1/002_coronal/",
        "/mnt/data/FA/200703_G23xM4/Fly1/003_coronal/",
        "/mnt/data/FA/200703_G23xM4/Fly1/005_coronal/",
        "/mnt/data/FA/200706_G23xM4/Fly1/002_coronal/",
        "/mnt/data/FA/200706_G23xM4/Fly1/003_coronal/",
        "/mnt/data/FA/200706_G23xM4/Fly2/002_coronal/",
    ]

    #experiments = [
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/001_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/002_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/003_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/004_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/005_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/006_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/007_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/008_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/009_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/010_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/011_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/012_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/013_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/014_coronal/",
    #    "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/015_coronal/",
    #]


    generators = []

    for experiment in experiments:
        red_stack_file = os.path.join(experiment, "2p/red.tif")
        full_red_stack = utils2p.load_img(red_stack_file)
        red_stack = full_red_stack.astype(np.uint16)[:200].copy()
        del full_red_stack
        green_stack_file = os.path.join(experiment, "2p/green.tif")
        full_green_stack = utils2p.load_img(green_stack_file)
        green_stack = full_green_stack.astype(np.uint16)[:200].copy()
        del full_green_stack
        generator = utils_video.generators.frames_2p(
            red_stack, green_stack, percentiles=(5, 99)
        )

        metadata_file = utils2p.find_metadata_file(experiment)
        metadata = utils2p.Metadata(metadata_file)
        power = metadata.get_power_reg1_start()
        gainA = metadata.get_gainA()
        gainB = metadata.get_gainB()
        text = f"Power: {power}, gainA: {gainA}, gainB: {gainB}"
        generator = utils_video.generators.add_text(generator, text, pos=(10, 30))
        # utils_video.make_video("parameters_2p.mp4", generator, 8)
        # exit()

        generators.append(generator)

    grid = utils_video.generators.grid(generators)
    utils_video.make_video("parameters_2p.mp4", grid, 15)


main()
