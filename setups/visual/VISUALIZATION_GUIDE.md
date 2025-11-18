# Hướng dẫn Visualize Attention của Model

Có nhiều cách để kiểm tra xem model đang tập trung vào đâu trong hình ảnh:

## 📋 Các phương pháp visualization

### 1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- **Cách hoạt động**: Sử dụng gradients để xác định vùng quan trọng
- **Ưu điểm**: Hoạt động với mọi model, không cần thay đổi architecture
- **File**: `visualize_simple.py`

### 2. **Patch Attention Maps**
- **Cách hoạt động**: Extract attention weights từ CLS token đến các patches
- **Ưu điểm**: Trực tiếp từ attention mechanism của Transformer
- **File**: `visualize_simple.py`

### 3. **Window-based MSA Attention**
- **Cách hoạt động**: Attention weights từ Window-based Multi-Head Self-Attention
- **Ưu điểm**: Chi tiết hơn, hiển thị attention trong từng window
- **File**: `visualize_attention.py`

### 4. **Channel Attention Maps**
- **Cách hoạt động**: Attention weights từ Channel Attention Block (CAB)
- **Ưu điểm**: Hiển thị kênh nào được model chú ý
- **File**: `visualize_attention.py`

### 5. **Reconstruction Attention**
- **Cách hoạt động**: Attention weights từ Dual Reconstruction module
- **Ưu điểm**: Hiển thị cách model reconstruct features
- **File**: `visualize_attention.py`

## 🚀 Cách sử dụng

### Script đơn giản (khuyến nghị)

```bash
python visualize_simple.py \
    --image_path /path/to/your/image.jpg \
    --model_path /path/to/checkpoint.pth \
    --arch vit_small \
    --patch_size 16 \
    --image_size 224 \
    --output_dir ./attention_viz
```

**Kết quả:**
- `gradcam_overlay.png`: Grad-CAM visualization
- `patch_attention_overlay.png`: Patch attention từ CLS token
- `patch_grid.png`: Grid visualization của patches

### Script đầy đủ (cho multi-attention architecture)

```bash
python visualize_attention.py \
    --image_path /path/to/your/image.jpg \
    --model_path /path/to/checkpoint.pth \
    --arch vit_small \
    --patch_size 16 \
    --image_size 224 \
    --use_mab True \
    --use_ocab True \
    --use_drff True \
    --output_dir ./attention_viz
```

## 📊 Giải thích kết quả

### Màu sắc trong heatmap:
- **🔴 Đỏ**: Vùng có attention cao (model tập trung nhiều)
- **🟡 Vàng**: Vùng có attention trung bình
- **🔵 Xanh**: Vùng có attention thấp (model ít chú ý)

### Các loại visualization:

1. **Original Image**: Hình ảnh gốc
2. **Attention Heatmap**: Bản đồ nhiệt của attention
3. **Overlay**: Kết hợp hình ảnh và heatmap để dễ nhìn

## 🔍 So sánh với mắt người

Để so sánh với cách bạn nhìn:

1. **Xem overlay visualization**: Màu đỏ = vùng model chú ý
2. **So sánh với vùng bạn nhìn**: 
   - Nếu trùng khớp → Model đang học đúng
   - Nếu khác → Có thể model đang học features khác

3. **Kiểm tra patch grid**: Xem patches nào được highlight

## 💡 Tips

1. **Thử nhiều hình ảnh**: Model có thể tập trung khác nhau với các loại ảnh khác nhau
2. **So sánh các layers**: Attention ở layers khác nhau có thể khác nhau
3. **Kiểm tra với support/query**: Trong few-shot learning, so sánh attention giữa support và query

## 🛠️ Troubleshooting

### Lỗi: "Cannot find model"
- Kiểm tra `--arch` có đúng không
- Đảm bảo model được import đúng trong `models/__init__.py`

### Lỗi: "Checkpoint not found"
- Kiểm tra đường dẫn checkpoint
- Đảm bảo checkpoint có keys: `params`, `state_dict`, hoặc trực tiếp là state_dict

### Attention map toàn màu xanh/đỏ
- Có thể do normalization, script sẽ tự động normalize
- Thử điều chỉnh `alpha` trong overlay

## 📝 Ví dụ output

```
[1/4] Loading image: test_image.jpg
   Image size: (224, 224)

[2/4] Loading model: vit_small
   Loading checkpoint: checkpoint.pth
   ✓ Model loaded successfully

[3/4] Computing Grad-CAM...
✓ Saved: ./attention_viz/gradcam_overlay.png
   Extracting patch attention...
✓ Saved: ./attention_viz/patch_attention_overlay.png
✓ Saved: ./attention_viz/patch_grid.png

[4/4] Visualization complete!
```

## 🎯 Next Steps

1. Chạy visualization trên nhiều hình ảnh
2. So sánh attention giữa các models khác nhau
3. Phân tích xem model có đang học đúng features không
4. Điều chỉnh training nếu cần

