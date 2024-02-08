#!/bin/sh

echo "CRITICAL: Your Python virtual environment enabled while running this script!"
echo "Current Python Environment: $(which python)"

# TODO: set your usename and API key here
export KAGGLE_USERNAME=YOUR KAGGLE USERNAME
export KAGGLE_KEY=YOUR KAGGLE KEY

ls | grep "data" &> /dev/null
if [ $? != 0  ]
then
    echo "ERROR: Cannot find data directory in current working directory $(pwd). Make sure to run this script in the root directory of the repository"
    exit 1
fi

# install the kaggle package
pip show kaggle &> /dev/null
if [ $? != 0 ]
then
    echo "Installing the kaggle package in Python environment"
    pip install -q kaggle
fi

mkdir -p data/ffhq
kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq -p data/ffhq

echo "Extracting dataset - this may take some time!"
unzip -d data/ffhq data/ffhq/flickrfaceshq-dataset-ffhq.zip
rm data/ffhq/flickrfaceshq-dataset-ffhq.zip
