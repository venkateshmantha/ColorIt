from os import listdir
from os.path import join, isfile, isdir
from typing import Tuple

from PIL import Image
from resizeimage import resizeimage

from dataset.shared import maybe_create_folder


class ImagenetResizer:

    def __init__(self, source_dir: str, dest_dir: str):
        if not isdir(source_dir):
            raise Exception('Input folder does not exist: {}'
                            .format(source_dir))
        self.source_dir = source_dir

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

    def resize_img(self, filename: str, size: Tuple[int, int] = (299, 299)):
        """
        Resizes the image using padding
        :param filename:
        :param size:
        """
        img = Image.open(join(self.source_dir, filename))
        original_width, original_height = img.size
        desired_width, desired_height = size

        ratio_w, ratio_h = desired_width / original_width, desired_height / original_height
        enlarge_factor = min(ratio_h, ratio_w)
        if enlarge_factor > 1:
            #enlarge the image in both directions to make that one fit
            enlarged_size = (int(original_width * enlarge_factor), int(original_height * enlarge_factor))
            img = img.resize(enlarged_size)

    
        res = resizeimage.resize_contain(img, size).convert('RGB')
        res.save(join(self.dest_dir, filename), res.format)

    def resize_all(self, size=(299, 299)):
        for filename in listdir(self.source_dir):
            if isfile(join(self.source_dir, filename)):
                self.resize_img(filename, size)


# Run as python3 -m dataset.resize <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals, dir_resized

    parser = argparse.ArgumentParser(
        description='Resize images from a folder to 299x299')
    parser.add_argument('-s', '--source-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='source')
    parser.add_argument('-o', '--output-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='output')

    args = parser.parse_args()
    ImagenetResizer(args.source, args.output).resize_all((299, 299))
