# Gait Graph

GAITGRAPH: GRAPH CONVOLUTIONAL NETWORK FOR SKELETON-BASED GAIT RECOGNITION
---

## Yêu cầu hệ thống
Có thể chạy các file ở local có cài python và các thư viện cơ bản: numpy, torch, pandas, ...
**Lưu ý:** Riêng file ipynb Train thì chạy ở google colab hoặc máy có gpu nhưng phải sửa các đường dẫn.

## Hướng dẫn sử dụng
Viết ở file HDSD_Code_Nhom6.docx

## Danh sách thư mục
```
GaitGraph/
├── data/                           # Chứa pose data của dataset CASIA-B     
├── images/                         # Chứa các hình ảnh về model
├── save/                           # Lưu file khi chạy train 
├── src/                            # Phần chính của dự án (chứa các notebook, model, ...)
│   ├── _pycache_                   # Thư mục lưu tự động sau khi chạy file
│   ├── _init_.py                   # File cần thiết để import các file python vào các notebook ipynb
│   ├── graph.py                    # Ma trận kề sử dụng trong chập đồ thị
│   ├── blocks.py                   # Các khối của model
│   ├── model_resgcn.py             # Model của dự án
│   ├── losses.py                   # Hàm mất mát
│   ├── data.py                     # Lấy và xử lý dữ liệu từ các file ở thư mục data
│   ├── augmentation.py             # Tăng cường dữ liệu
│   ├── 0_Check_data.ipynb          # Notebook để kiểm tra dữ liệu sau tăng cường
│   ├── 1_Train.ipynb               # Notebook train model
│   ├── 2_Evaluate_model.ipynb      # Notebook evaluate model
│   ├── 3_Embedding_vector.ipynb    # Notebook đánh giá model bằng hiển thị trực quan  
└── README.md                       # Hướng dẫn chi tiết
└── HDSD_Code_Nhom6.docx            # Hướng dẫn sử dụng code
```