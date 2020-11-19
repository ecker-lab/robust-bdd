

from copy import copy
import os
import urllib.request
from tqdm import tqdm


DL_LINKS = {
    "table_1_plain": ["https://data.goettingen-research-online.de/api/access/datafile/20240?gbrecs=true","twostream-resnet50_all_plain.pt"],
    "table_2_plain": ["https://data.goettingen-research-online.de/api/access/datafile/20243?gbrecs=true","twostream-resnet50_gupta_plain.pt"],
    "table_3_plain_1": ["https://data.goettingen-research-online.de/api/access/datafile/20247?gbrecs=true","twostream-resnet50_ood1_plain.pt"],
    "table_3_plain_2": ["https://data.goettingen-research-online.de/api/access/datafile/20251?gbrecs=true","twostream-resnet50_ood2_plain.pt"],
    "table_3_plain_3": ["https://data.goettingen-research-online.de/api/access/datafile/20255?gbrecs=true","twostream-resnet50_ood3_plain.pt"],
    "table_4_plain_g": ["https://data.goettingen-research-online.de/api/access/datafile/20243?gbrecs=true","twostream-resnet50_gupta_plain.pt"],
    "table_4_plain_1": ["https://data.goettingen-research-online.de/api/access/datafile/20247?gbrecs=true","twostream-resnet50_ood1_plain.pt"],
    "table_4_plain_2": ["https://data.goettingen-research-online.de/api/access/datafile/20251?gbrecs=true","twostream-resnet50_ood2_plain.pt"],
    "table_4_plain_3": ["https://data.goettingen-research-online.de/api/access/datafile/20255?gbrecs=true","twostream-resnet50_ood3_plain.pt"],
    "table_4_swa_g": ["https://data.goettingen-research-online.de/api/access/datafile/20244?gbrecs=true","twostream-resnet50_gupta_swa.pt"],
    "table_4_swa_1": ["https://data.goettingen-research-online.de/api/access/datafile/20248?gbrecs=true","twostream-resnet50_ood1_swa.pt"],
    "table_4_swa_2": ["https://data.goettingen-research-online.de/api/access/datafile/20252?gbrecs=true","twostream-resnet50_ood2_swa.pt"],
    "table_4_swa_3": ["https://data.goettingen-research-online.de/api/access/datafile/20256?gbrecs=true","twostream-resnet50_ood3_swa.pt"],
    "table_4_multiadabn_g": ["https://data.goettingen-research-online.de/api/access/datafile/20241?gbrecs=true","twostream-resnet50_gupta_multi.pt"],
    "table_4_multiadabn_1": ["https://data.goettingen-research-online.de/api/access/datafile/20245?gbrecs=true","twostream-resnet50_ood1_multi.pt"],
    "table_4_multiadabn_2": ["https://data.goettingen-research-online.de/api/access/datafile/20249?gbrecs=true","twostream-resnet50_ood2_multi.pt"],
    "table_4_multiadabn_3": ["https://data.goettingen-research-online.de/api/access/datafile/20253?gbrecs=true","twostream-resnet50_ood3_multi.pt"],
    "table_4_multiadabnswa_g": ["https://data.goettingen-research-online.de/api/access/datafile/20242?gbrecs=true","twostream-resnet50_gupta_multiswa.pt"],
    "table_4_multiadabnswa_1": ["https://data.goettingen-research-online.de/api/access/datafile/20246?gbrecs=true","twostream-resnet50_ood1_multiswa.pt"],
    "table_4_multiadabnswa_2": ["https://data.goettingen-research-online.de/api/access/datafile/20250?gbrecs=true","twostream-resnet50_ood2_multiswa.pt"],
    "table_4_multiadabnswa_3": ["https://data.goettingen-research-online.de/api/access/datafile/20254?gbrecs=true","twostream-resnet50_ood3_multiswa.pt"],
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights(setting_name):
    #TODO Put here the right setting -> downloadpath conversion
    dl_path, outfile = DL_LINKS[setting_name]
    filepath = "../weights/"+outfile
    print("Downloading from {} to {}".format(dl_path, filepath))
    if not os.path.isfile(filepath):
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dl_path.split('/')[-1]) as t:
            urllib.request.urlretrieve(dl_path, filename = filepath, reporthook=t.update_to)
        print("Downloaded!")
    else:
        print("File existed allready!")
    return filepath
