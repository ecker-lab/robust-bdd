"""Helper Functions for cutting and resizing images.
"""

from PIL import Image

IMG_FOLDER = '../train/images/'
IMG_PRE_FILE = 'hurricane-harvey_00000114_pre_disaster.png'
IMG_POST_FILE = 'hurricane-harvey_00000114_post_disaster.png'
LABEL_FOLDER = '../train/labels/'
LABEL_FILE = 'hurricane-harvey_00000114_post_disaster.json'
MASK_FOLDER = '../trialout/mask/'
MASK_FILE = 'hurricane-harvey_00000114_mask.png'
OUTPUT_FOLDER_IMAGES = '../trialout/resize/images/'
OUTPUT_FOLDER_MASKS = '../trialout/resize/masks/'

def cut_and_resize(IMAGE_FOLDER,IMAGE_PRE_FILE,IMAGE_POST_FILE,MASK_FOLDER,MASK_FILE,OUTPUT_FOLDER_IMAGES,OUTPUT_FOLDER_MASKS,cutwidth,resizewidth):
    """
    Cuts square input Image and Mask into (image width/cutwidth)**2 tiles and resizes each of them to size resizewidthxresizewidth
    """
    Img_Pre = Image.open(IMAGE_FOLDER+IMAGE_PRE_FILE)
    Img_Post = Image.open(IMAGE_FOLDER+IMAGE_PRE_FILE)
    Mask = Image.open(MASK_FOLDER+MASK_FILE)
    imgwidth, imgheight = Img_Pre.size
    print(imgwidth/cutwidth)
    for i in range(0,imgheight,cutwidth):
        for j in range(0,imgwidth,cutwidth):
            box = (j,i,j+cutwidth,i+cutwidth)
            Img_Pre_crop = Img_Pre.crop(box)
            Img_Post_crop = Img_Post.crop(box)
            Mask_crop = Mask.crop(box)
            newsize = (resizewidth,resizewidth)
            Img_Pre_resize = Img_Pre_crop.resize(newsize)
            Img_Post_resize = Img_Post_crop.resize(newsize)
            Mask_resize = Mask_crop.resize(newsize)
            num = int(imgwidth/cutwidth**2*j+i/cutwidth)
            Img_Pre_resize.save(OUTPUT_FOLDER_IMAGES+IMAGE_PRE_FILE[:-4]+"_Tile_%s.png" % num)
            Img_Post_resize.save(OUTPUT_FOLDER_IMAGES+IMAGE_POST_FILE[:-4]+"_Tile_%s.png" % num)
            Mask_resize.save(OUTPUT_FOLDER_MASKS+MASK_FILE[:-4]+"_Tile_%s.png" % num)

def cut_and_resize_one(IN_FOLDER,FILE,OUT_FOLDER,cutwidth,resizewidth):
    """
    Cuts square Input into (image width/cutwidth)**2 tiles and resizes each of them to size resizewidthxresizewidth
    """
    Img = Image.open(IN_FOLDER+FILE)
    imgwidth, imgheight = Img.size
    if imgheight == cutwidth:
        newsize = (resizewidth,resizewidth)
        Img_resize = Img.resize(newsize)
        Img_resize.save(OUT_FOLDER+FILE)

    else:
        for i in range(0,imgheight,cutwidth):
            for j in range(0,imgwidth,cutwidth):
                box = (j,i,j+cutwidth,i+cutwidth)
                Img_crop = Img.crop(box)
                newsize = (resizewidth,resizewidth)
                Img_resize = Img_crop.resize(newsize)
                num = int(imgwidth/cutwidth**2*j+i/cutwidth)
                Img_resize.save(OUT_FOLDER+FILE[:-4]+"_Tile_%s.png" % num)



if __name__ == "__main__":
    cut_and_resize(IMG_FOLDER,IMG_PRE_FILE,IMG_POST_FILE,MASK_FOLDER,MASK_FILE,OUTPUT_FOLDER_IMAGES,OUTPUT_FOLDER_MASKS,256,32)
