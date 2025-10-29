#!/bin/bash
# Download and extract FC100 dataset using gdown

# Tải file từ Google Drive
gdown "https://drive.google.com/uc?id=1nEh3O2RJ5zTbWj7luyQCNcklpX0_5KlS" -O FC100.tar

# Giải nén
tar -xvf FC100.tar FC100/

echo "✅ FC100 dataset downloaded and extracted successfully!"


