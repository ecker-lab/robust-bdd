

from copy import copy
import os
import urllib.request

DL_LINKS = {
    "table_1_plain": ["","twostream-resnet50_table_1_plain.pt"]
}


def download_weights(setting_name):
    #TODO Put here the right setting -> downloadpath conversion
    dl_name, outfile = DL_LINKS[setting_name]
    base_path = "https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/TRTODX"
    dl_path = base_path+"/"+dl_name
    if not os.path.isfile("weights/"+outfile):
        urllib.request.urlretrieve(dl_path, "weights/"+outfile)
    return "weights/"+outfile
