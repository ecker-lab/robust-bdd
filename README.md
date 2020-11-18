# robust-bdd
Code for Benson &amp; Ecker (2020): Assessing out-of-domain generalization for robust building damage detection.


# Setup
Run ´bash -i setup.sh /path/to/xbd/´.
This will install a conda environment ´robust-bdd´ and prepare the xBD data for the two-stream ResNet50.
Download https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w with your web-browser and move the file to ´robust-bdd/dualhrnet´. These are the ImageNet-pretrained weights for HRNetv2.