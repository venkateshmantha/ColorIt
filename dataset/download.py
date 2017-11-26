import urllib.request, hashlib, imghdr, sys, tarfile
from itertools import islice
from os.path import join, isfile
from typing import Union, List

from dataset.shared import dir_root, maybe_create_folder


class ImagenetDownloader:
    def __init__(self, links_source: str, dest_dir: str):

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

        if not isfile(links_source):
            raise Exception('Links source not valid: {}'.format(links_source))

        if links_source.endswith('.tgz'):
            with tarfile.open(links_source, 'r:gz') as tar:
                tar.extractall(path=dir_root)
                links_source = join(dir_root, 'fall11_urls.txt')

        self.links_source = links_source

    def _download_img(self, image_url: str) -> Union[str, None]:
        image_name = self._encode_image_name(image_url)
        image_path = join(self.dest_dir, image_name)
        if not isfile(image_path):
            try:
                request = urllib.request.urlopen(image_url, timeout=2)
                image = request.read()
                if imghdr.what('', image) == 'jpeg':
                    with open(image_path, "wb") as f:
                        f.write(image)
            except Exception as e:
                print('Error downloading {}: {}'.format(image_url, e),
                      file=sys.stderr)
                return None
        return image_path

    def download_images(self, size: int = 10, skip: int = 0) -> List[str]:
        urls = islice(self._image_urls_generator(), skip, skip + size)
        downloaded_images = map(self._download_img, urls)
        valid_images = filter(lambda x: x is not None, downloaded_images)
        return list(valid_images)

    @staticmethod
    def _encode_image_name(image_url: str) -> str:
        return hashlib.md5(image_url.encode('utf-8')).hexdigest() + '.jpeg'



# Run as python3 -m dataset.download <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals

    links_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Download and process images from imagenet')
    parser.add_argument('-c', '--count',
                        default=10,
                        type=int)
    parser.add_argument('--skip',
                        default=0,
                        type=int,
                        metavar='N')
    parser.add_argument('-s', '--source',
                        default=links_url,
                        type=str,
                        dest='source')
    parser.add_argument('-o', '--output-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='output')

    args = parser.parse_args()
    ImagenetDownloader(args.source, args.output) \
        .download_images(args.count, args.skip)
