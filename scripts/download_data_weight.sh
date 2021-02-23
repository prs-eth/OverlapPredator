wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/Predator/predator.zip
unzip predator.zip
mv predator/data .
mv predator/weights .
rm -r predator
rm predator.zip
