import imageio
import sys
import datetime

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import os

    current_dir = os.getcwd()
    path = os.path.join(current_dir, 'results')
    filenames = [f for f in listdir(path) if isfile(join(path, f))]

    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)

    filenames = [os.path.join(path, f) for f in filenames]
    filenames.sort(key=lambda x: x)
    print(filenames)
    create_gif(filenames, 4/30)
