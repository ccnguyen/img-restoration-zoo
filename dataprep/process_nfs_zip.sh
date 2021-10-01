#!/bin/bash
# Download the Need for Speed dataset via `curl -fSsl http://ci2cv.net/nfs/Get_NFS.sh | bash -`
# Unzip the zip file
# This script processes the data from this unzipping step

cd ../data/nfs
# unzip all movies
unzip \*.zip
# remove zip files
rm *.zip
# remove all 30fps data
find . -name 30 -type d -exec rm -r {} +