#!/bin/bash
# Download and extract CIFAR-FS dataset using gdown

# Tải dataset
gdown "https://drive.google.com/uc?id=1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8" -O cifar_fs.tar

# Giải nén
tar -xvf cifar_fs.tar cifar_fs/

echo "✅ CIFAR-FS dataset downloaded and extracted successfully!"
