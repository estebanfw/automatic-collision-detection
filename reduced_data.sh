#!/bin/bash
echo "Downloading data from internet"
wget https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip
echo "Extracting data from zip"
unzip train_data.zip
echo "Moving data to data/"
mv train_data.csv data/
echo "For testing purposes only a part of the dataframe will be used"
ls -lh data/
head --lines 20000 data/train_data.csv > data/aux.csv
mv data/aux.csv data/train_data.csv
wc -l data/train_data.csv
ls -lh 
ls -lh data/    