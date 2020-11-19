# Assessing out-of-domain generalization for robust building damage detection
Code for Benson &amp; Ecker (2020): Assessing out-of-domain generalization for robust building damage detection.


# Setup
Get yourself a copy of the xBD dataset from https://xView2.org

Install Anaconda or Miniconda.

Run 
```
bash -i setup.sh /path/to/xbd/
```
This will install a conda environment `robust-bdd` and prepare the xBD data for the two-stream ResNet50.

Download https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w with your web-browser and move the file to `robust-bdd/dualhrnet`.

These are the ImageNet-pretrained weights for HRNetv2 coming from https://github.com/HRNet/HRNet-Image-Classification.

# Use
*Note: All of the following requires a GPU. Our experiments were performed either with 2 GTX1080 or with 1 RTX 2080TI.*

You can re-train the models presented in the Paper by running:
```
bash train.sh path/to/xbd/ <Table number> <twostream-resnet50 or dualhrnet> <plain or multi>
```
You can reproduce the inference by the models run in the Paper with:
```
bash test.sh path/to/xbd/ <Table number> <twostream-resnet50 or dualhrnet> <plain or swa or multiadabn or multiadabnswa>
```
# Weights
Pre-trained weights are automatically downloaded when using the above `test.sh` script.

If you want to manually download them, find them under:

Benson, Vitus, 2020, "Replication Data for: Assessing out-of-domain generalization for robust building damage detection", https://doi.org/10.25625/TRTODX, Göttingen Research Online / Data, V2
