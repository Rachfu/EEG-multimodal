#!/bin/bash

echo "<<<<<<<<<<<<<<< data processing... >>>>>>>>>>>>>>>"
python src/data/process.py

if [ $? -eq 0 ]; then
    echo "src/data/process.py executed successfully; results in data/processed/"
else
    echo "src/data/process.py failed to execut"
    exit 1
fi


echo "<<<<<<<<<<<<<<< get embedding... >>>>>>>>>>>>>>>"
python src/data/get_embedding.py

if [ $? -eq 0 ]; then
    echo "src/data/get_embedding.py executed successfully; results in data/embedding/"
else
    echo "src/data/get_embedding.py failed to execut"
    exit 1
fi

