#!/bin/bash

cd /media/data6/cindy/data/GOPRO_Large_all
mkdir train
mkdir test

while read file; do mv "$file" /media/data6/cindy/data/GOPRO_Large_all/train; done < /home/cindy/img-restoration-zoo/dataprep/gopro_train.txt

while read file; do mv "$file" /media/data6/cindy/data/GOPRO_Large_all/test; done < /home/cindy/img-restoration-zoo/dataprep/gopro_test.txt

