# 6451071013_Lab23 – Khai phá dữ liệu, Thực hành buổi 3

**Sinh viên:** Mai Quốc Đại – **MSSV:** 6451071013 – **Lớp:** CQ.64.CNTT
**GVHD:** ThS. Nguyễn Thiện Dương

| File | Nội dung |
|---|---|
| `6451071013_LAB3_Bai1.ipynb` | Bài 1 (trang 1): Iris – Decision Tree, Naïve Bayes, KNN, Neural Network |
| `6451071013_LAB3_Bai2.ipynb` | Bài 2 (trang 5): MNIST – Neural Network, chia train/validation/test |
| `6451071013_LAB3_Bai3.ipynb` | Bài 3 (trang 7): MNIST – Neural Network, 10-fold cross validation |
| `6451071013_LAB3_Bai4.ipynb` | Bài 4 (trang 9): Iris – Neural Network, 10-fold cross validation |
| `6451071013_LAB3_Bai5.ipynb` | Bài 5 (trang 9–11): Neural Network dự đoán doanh thu game |
| `6451071013_LAB3_Bai2_Phan2.ipynb` | Bài 2 (trang 18–19): Gom cụm Mall Customers bằng k-Means |
| `6451071013_LAB3_Bai5_Phan2.ipynb` | Bài 5 (trang 26): Gom cụm MNIST bằng k-Means, so sánh với nhãn |
| `6451071013_LAB3_Bai6_Phan2.ipynb` | Bài 6 (trang 26): Giảm số màu ảnh bằng k-Means |
| `6451071013_LAB3_Bai5_BaiTapThem.ipynb` | Bài 5 (trang 29): Gom cụm Breast Cancer Wisconsin |
| `6451071013_LAB3_BaoCao.docx` | Báo cáo theo template (mã nguồn + hình kết quả) |

## Dữ liệu (`data/`)
- `Mall_Customers.csv`, `breast_cancer/wdbc.data`: từ thư mục Google Drive của môn học.
- `sales_data_training.csv`, `sales_data_test.csv`: bộ dữ liệu doanh thu game (khóa Keras của LinkedIn Learning).
- `image.png`: ảnh mẫu `china.jpg` của scikit-learn, dùng cho bài giảm số màu.
- Iris và MNIST được tải trực tiếp từ sklearn/Keras.

## Chạy lại
```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn plotly pillow nbclient python-docx
python tools/build_notebooks.py   # sinh lại các notebook từ mã nguồn
python tools/run_all.py           # chạy tất cả notebook, lưu output
python tools/build_report.py      # tạo báo cáo Word
```
