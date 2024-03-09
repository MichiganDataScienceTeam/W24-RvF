#!/bin/bash

if [ $1 == "rvf10k" ]
then
    DATASET_NAME="rvf10k"
    DATASET_OWNER="sachchitkunichetty"
    DATASET_ID="rvf10k"
elif [ $1 == "rvf140k" ]
then
    DATASET_NAME="rvf140k"
    DATASET_OWNER="xhlulu"
    DATASET_ID="140k-real-and-fake-faces"
else
    echo "ERROR: Invalid dataset name: \"$1\". Expected either \"rvf10k\" or \"rvf140k\""
    exit 1
fi

function download() {
    echo "Downloading $DATASET_NAME to data/$DATASET_NAME"
    kaggle datasets download -d $DATASET_OWNER/$DATASET_ID -p data/$DATASET_NAME
   
}

function clean () {
    if [ $DATASET_NAME == "rvf10k" ]
    then
        mv data/$DATASET_NAME/rvf10k/train data/$DATASET_NAME/train
        mv data/$DATASET_NAME/rvf10k/valid data/$DATASET_NAME/valid
        
        rmdir data/$DATASET_NAME/rvf10k
    else [ $DATASET_NAME == "rvf140k" ]
        mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/train data/$DATASET_NAME/train
        mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/valid data/$DATASET_NAME/valid
        mv data/$DATASET_NAME/real_vs_fake/real-vs-fake/test data/$DATASET_NAME/test

        rmdir data/$DATASET_NAME/real_vs_fake/real-vs-fake
        rmdir data/$DATASET_NAME/real_vs_fake
    fi
}

echo "WARNING: Your Python virtual environment should be enabled while running this script!"
echo "Current Python Environment: $(which python)"

if [ -z "${KAGGLE_CONFIG_DIR}" ]; then
    export KAGGLE_CONFIG_DIR=secrets
fi

if [ -d "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "ERROR: Cannot find kaggle.json in $KAGGLE_CONFIG_DIR"
    exit 1
fi

if ! [ -d "data" ]; then
    echo "ERROR: Cannot find data directory in current working directory $(pwd). Make sure to run this script in the root directory of the repository"
    exit 1
fi

# install the kaggle package
pip show kaggle &>/dev/null
if [ $? != 0 ]; then
    echo "Installing the kaggle package in Python environment"
    pip install -q kaggle
fi

count=$(ls data/$DATASET_NAME | wc -l)

# download and extract the dataset
if ! [ -d data/$DATASET_NAME ] || [ $count = "0" ]; then
    mkdir -p data/$DATASET_NAME
    download

    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/$DATASET_ID.zip -d data/$DATASET_NAME

    clean
elif [ $count = "1" ] && [ -f data/$DATASET_NAME/$DATASET_ID.zip ]
then
    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/$DATASET_ID.zip -d data/$DATASET_NAME

    clean
elif [ -d data/$DATASET_NAME/real_vs_fake ] || [ -d data/$DATASET_NAME/rvf10k  ]
then
    clean
else
    echo "Using cached version of $DATASET_NAME"
fi
