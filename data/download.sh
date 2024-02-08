#!/bin/sh

DATASET_NAME="rvf140k"

function download () {
    echo "Downloading $DATASET_NAME to data/$DATASET_NAME"
    kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/$DATASET_NAME
   
}

function clean () {
    mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/train data/$DATASET_NAME/train
    mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/valid data/$DATASET_NAME/valid
    mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/test data/$DATASET_NAME/test

    rmdir data/$DATASET_NAME/real_vs_fake/real-vs-fake
    rmdir data/$DATASET_NAME/real_vs_fake
}

echo "CRITICAL: Your Python virtual environment enabled while running this script!"
echo "Current Python Environment: $(which python)"

# TODO: set the directory containing your credentials below:
export KAGGLE_CONFIG_DIR=secrets

if [ -d "$KAGGLE_CONFIG_DIR/kaggle.json" ]
then
    echo "ERROR: Cannot find kaggle.json in $KAGGLE_CONFIG_DIR"
    exit 1
fi

if ! [ -d "data"  ]
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

count=$(ls data/$DATASET_NAME | wc -l)

# download and extract the dataset
if ! [ -d data/$DATASET_NAME ] || [ $count = "0" ]
then
    mkdir -p data/$DATASET_NAME
    download

    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/140k-real-and-fake-faces.zip -d data/$DATASET_NAME

    clean
elif [ $count = "1" ] && [ -f data/$DATASET_NAME/140k-real-and-fake-faces.zip ]
then
    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/140k-real-and-fake-faces.zip -d data/$DATASET_NAME

    clean
elif [ -d data/$DATASET_NAME/real_vs_fake ]
then
    clean
else
    echo "Using cached version of $DATASET_NAME"
fi


