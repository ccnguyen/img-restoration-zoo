#!/bin/bash

cd /media/data4b/cindy_data/data/GOPRO_Large_all
mkdir train
mkdir test

while read file; do mv "$file" /media/data4b/cindy_data/data/GOPRO_Large_all/train; done < /home/cindy/home/cindy/stor/cindy/coded-deblur/dataprep/gopro_train.txt

while read file; do mv "$file" /media/data4b/cindy_data/data/GOPRO_Large_all/test; done < /home/cindy/home/cindy/stor/cindy/coded-deblur/dataprep/gopro_test.txt

