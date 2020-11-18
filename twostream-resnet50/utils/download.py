

from copy import copy
import os
import urllib.request
from tqdm import tqdm


DL_LINKS = {
    "table_1_plain": ["https://data.goettingen-research-online.de/api/access/datafile/20240?gbrecs=true","twostream-resnet50_all_plain.pt"]
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights(setting_name):
    #TODO Put here the right setting -> downloadpath conversion
    dl_path, outfile = DL_LINKS[setting_name]
    filepath = "../weights2/"+outfile
    print("Downloading from {} to {}".format(dl_path, filepath))
    if not os.path.isfile(filepath):
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dl_path.split('/')[-1]) as t:
            urllib.request.urlretrieve(dl_path, filename = filepath, reporthook=t.update_to)
        print("Downloaded!")
    else:
        print("File existed allready!")
    return filepath
