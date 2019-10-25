#!/bin/bash

source activate skypool_lmc

echo "Transform data to coco format ..."
python ./code/Fabric2COCO.py
echo "OK ..."
