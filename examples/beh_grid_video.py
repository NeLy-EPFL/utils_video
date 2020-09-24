import os.path

import utils_video
import utils_video.generators


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

    #experiments = [
    #    "/mnt/data/FA/200703_G23xM4/Fly1/001_coronal/",
    #    "/mnt/data/FA/200703_G23xM4/Fly1/002_coronal/",
    #    "/mnt/data/FA/200703_G23xM4/Fly1/003_coronal/",
    #    "/mnt/data/FA/200703_G23xM4/Fly1/004_coronal/",
    #    "/mnt/data/FA/200703_G23xM4/Fly1/005_coronal/",
    #    "/mnt/data/FA/200706_G23xM4/Fly1/001_coronal/",
    #    "/mnt/data/FA/200706_G23xM4/Fly1/002_coronal/",
    #    "/mnt/data/FA/200706_G23xM4/Fly1/003_coronal/",
    #    "/mnt/data/FA/200706_G23xM4/Fly2/001_coronal/",
    #    "/mnt/data/FA/200706_G23xM4/Fly2/002_coronal/",
    #]

    experiments = [
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/001_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/002_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/003_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/004_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/005_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/006_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/007_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/008_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/009_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/010_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/011_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/012_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/013_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/014_coronal/",
        "/mnt/data/FA/200707_ABO_GC7c_tdTom/Fly1/015_coronal/",
    ]


    generators = []

    for experiment in experiments:
        #images_path = os.path.join(experiment, "behData/images/camera_1_img_*.jpg")
        #generator = utils_video.generators.images(images_path)
        video_path = os.path.join(experiment, "behData/images/camera_1.mp4")
        generator = utils_video.generators.video(video_path)

        generators.append(generator)

    grid = utils_video.generators.grid(generators)
    utils_video.make_video("beh_grid.mp4", grid, 100, n_frames=6000)


main()
