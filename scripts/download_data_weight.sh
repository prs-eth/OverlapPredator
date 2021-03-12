wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/Predator/predator.zip
unzip predator.zip
mv predator/data .
mv predator/weights .
rm -r predator
rm predator.zip
cp assets/cloud_bin_21.pth data/indoor/test/7-scenes-redkitchen
cp assets/cloud_bin_34.pth data/indoor/test/7-scenes-redkitchen
