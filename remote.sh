#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
.git/*
logs/*
outputs/*
weights/*
notebooks/*
data/*
.mypy_cache/*
*.o
*.so
*.flac
*.mp4
*.jpg
*.tar
*.dcm
*.mp3
*.dicom
*.xml
*.swp
*.pb
*.tmp
*.onnx
*.zip
*.parquet
# *.html
*.ipynb
*.feather
*.pth
*.wav
*.pkl
*.npy
*.jpg
*.zip
EOM

if [ "$1" == "superbot" ]; then
    echo "Push code to superbot"
    IP="superbot.us-central1-a.s-307909"
    REMOTE_HOME="/home/anhph17"
elif [ "$1" == "a100x16-qa" ]; then
    echo "h2-hs-us-a100x16-instance-9"
    IP="h2-hs-us-a100x16-instance-9.us-central1-f.ai-sekisan-qa"
    REMOTE_HOME="/home/anhph"
elif [ "$1" == "h100" ]; then
    echo "h100"
    IP="h100"
    REMOTE_HOME="/home/ubuntu/anhpham"
elif [ "$1" == "a100" ]; then
    echo "a100"
    IP="a100"
    REMOTE_HOME="/home/ubuntu/anhpham"
elif [ "$1" == "dev" ]; then
    echo "dev"
    IP="dev"
    REMOTE_HOME="/home/ubuntu/anhpham"
elif [ "$1" == "dev_2" ]; then
    echo "dev_2"
    IP="dev_2"
    REMOTE_HOME="/home/ubuntu/anhpham"
elif [ "$1" == "dev_h100" ]; then
    echo "dev_h100"
    IP="dev_h100"
    REMOTE_HOME="/home/ubuntu/anhpham"
elif [ "$1" == "dev_h100_2" ]; then
    echo "dev_h100_2"
    IP="dev_h100_2"
    REMOTE_HOME="/home/andrew"
elif [ "$1" == "dev_h100_dp" ]; then
    echo "dev_h100_dp"
    IP="dev_h100_dp"
    REMOTE_HOME="/home/andrew"
elif [ "$1" == "dev_h100_dp_102" ]; then
    echo "dev_h100_dp_102"
    IP="dev_h100_dp_102"
    REMOTE_HOME="/home/andrew"
else
    echo "Unknown instance"
    exit
fi

# IP=$1
# echo "Push code to $IP"
# REMOTE_HOME="/home/phamhoan"

# push code to server
rsync -vr -P -e "ssh" --exclude-from $TEMP_FILE "$PWD" $IP:$REMOTE_HOME/

# remove temp. file
rm $TEMP_FILE
