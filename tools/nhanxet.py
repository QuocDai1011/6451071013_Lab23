"""Cac doan nhan xet (markdown) chen vao notebook, viet dua tren ket qua da chay."""

NHANXET = {
"NHANXET_BAI1": """**Nhận xét:**

- Trên tập test (30 mẫu, `random_state=42`) cả 4 mô hình Decision Tree, Naïve Bayes, KNN và Neural Network đều đạt Accuracy, Precision, Recall, F1 = 1.0. Iris là bộ dữ liệu nhỏ, các lớp gần như tách biệt tuyến tính theo chiều dài/rộng cánh hoa, nên mô hình nào cũng phân lớp đúng toàn bộ.
- Vì tập test quá nhỏ (mỗi mẫu sai làm giảm ~3.3% accuracy), kết quả 100% chưa đủ để kết luận mô hình nào tốt hơn. Kiểm định chéo 10-fold ở trên cho thấy các mô hình đều đạt khoảng 93–97%; lớp setosa luôn được phân lớp đúng, sai sót chủ yếu nằm giữa versicolor và virginica (hai lớp có vùng giá trị chồng lấn).
- Neural Network cần nhiều thời gian huấn luyện nhất (100 epoch, batch_size = 1) nhưng không cho kết quả vượt trội trên dữ liệu nhỏ này; Naïve Bayes và KNN đơn giản, huấn luyện tức thì mà vẫn chính xác. Decision Tree dễ diễn giải nhưng dễ overfit khi dữ liệu lớn hơn.
- Ở câu i, cả 4 mô hình dự đoán giống nhau: [6.2, 2.9, 4.3, 1.3] → versicolor, [5.1, 3.5, 1.4, 0.2] → setosa, [7.3, 2.8, 6.4, 2.1] → virginica.""",

"ELBOW1": """**Chọn k:** Inertia giảm mạnh từ k = 1 đến k = 4 (171 535 → 28 166), sau đó chỉ giảm nhẹ (k = 5: 23 812). Điểm khuỷu tay ở **k = 4**.""",
"ELBOW2": """**Chọn k:** Inertia giảm mạnh đến k = 5 (44 449), từ k = 6 trở đi mức giảm nhỏ dần (37 234, 30 241 …). Điểm khuỷu tay ở **k = 5**.""",
"ELBOW3": """**Chọn k:** Đường cong giảm đều hơn; mức giảm rõ cuối cùng là từ k = 5 (75 351) xuống k = 6 (58 300), sau đó chỉ còn giảm khoảng 7 000 mỗi bước. Chọn **k = 6**.""",

"NHANXET_MALL": """**Nhận xét:**

- Silhouette: (Tuổi, Điểm mua sắm) k = 4 → 0.4997; (Thu nhập, Điểm mua sắm) k = 5 → **0.5539** (tốt nhất); (Tuổi, Thu nhập, Điểm mua sắm) k = 6 → 0.4523.
- Cặp Thu nhập – Điểm mua sắm cho các cụm tách biệt rõ nhất, gồm 5 nhóm khách hàng dễ diễn giải: thu nhập cao – chi tiêu cao (~86k$, điểm 82), thu nhập cao – chi tiêu thấp (~88k$, điểm 17), thu nhập thấp – chi tiêu cao (~26k$, điểm 79), thu nhập thấp – chi tiêu thấp (~26k$, điểm 21) và nhóm trung bình (~55k$, điểm 50).
- Cặp Tuổi – Điểm mua sắm cho thấy khách hàng trẻ (khoảng 30 tuổi) có xu hướng chi tiêu nhiều, còn nhóm trên 40 tuổi phần lớn chi tiêu ở mức thấp hoặc trung bình.
- Gom cụm 3 thuộc tính chia nhỏ hơn (6 nhóm) nhưng silhouette thấp hơn, do thêm chiều Tuổi làm các cụm chồng lấn nhiều hơn.""",

"NHANXET_MNIST_KM": """**Nhận xét:**

- k-Means (k = 10) gom cụm dựa trên khoảng cách Euclid giữa các vector pixel, không dùng nhãn. Sau khi gán mỗi cụm với chữ số chiếm đa số, độ chính xác chỉ khoảng **59%** trên cả train và test (ARI ≈ 0.37, NMI ≈ 0.50), thấp hơn nhiều so với Neural Network có giám sát ở bài trước (~97%).
- Các chữ số có hình dạng đặc trưng như 0, 1, 2, 6 tạo thành cụm khá thuần khiết (độ thuần khiết 78–91%). Ngược lại, 4 – 7 – 9 bị gộp chung vào các cụm 6 và 7, còn 3 – 5 – 8 bị gộp vào cụm 2 và 8.
- Chữ số 1 và 0 mỗi chữ chiếm 2 cụm (do có nhiều kiểu viết: nét thẳng và nét nghiêng), nên không còn cụm nào đại diện cho chữ số 5 và 9, khiến recall của hai lớp này bằng 0.
- Kết luận: k-Means trên pixel thô phản ánh được một phần cấu trúc dữ liệu, nhưng không thay thế được mô hình phân lớp có giám sát.""",

"NHANXET_IMG": """**Nhận xét:**

- Ảnh gốc có 96 615 màu khác nhau. Sau khi gom cụm về 32 màu, ảnh gần như không khác biệt khi nhìn bằng mắt thường, chỉ các vùng chuyển màu mịn (bầu trời) xuất hiện vài dải màu.
- Lưu dạng RGB thông thường, file giảm từ ~470 KB xuống ~263 KB. Khi lưu dưới dạng ảnh bảng màu (palette, chỉ cần 5 bit cho mỗi pixel vì có 32 màu), file chỉ còn ~75 KB, khoảng 16% kích thước ban đầu.
- Số màu k càng nhỏ thì file càng nhẹ nhưng ảnh càng mất chi tiết (4 – 8 màu thấy rõ các mảng màu). Có thể chọn k để cân bằng giữa chất lượng ảnh và kích thước file mong muốn.""",

"NHANXET_BC": """**Nhận xét:**

- Theo silhouette, k = 2 là số cụm tốt nhất (0.3434), phù hợp với hai nhóm lành tính / ác tính của bài toán.
- k-Means với k = 2 (không dùng nhãn khi huấn luyện) cho kết quả khớp với chẩn đoán thực tế ở mức **Accuracy ≈ 90.5%** (ARI ≈ 0.65). Lớp lành tính có recall 0.95, lớp ác tính có recall 0.83: khoảng 17% khối u ác tính bị xếp vào cụm lành tính.
- Trên biểu đồ PCA, hai nhóm tách khá rõ dọc theo trục PC1 (liên quan đến kích thước nhân tế bào như radius, perimeter, area), nhưng vùng giáp ranh có chồng lấn nên k-Means không phân tách được hoàn toàn.
- Trong y tế, bỏ sót ca ác tính là sai lầm nghiêm trọng. Gom cụm hữu ích để khám phá cấu trúc dữ liệu, nhưng chẩn đoán thực tế cần mô hình có giám sát và chuyên gia kiểm tra.""",
}
