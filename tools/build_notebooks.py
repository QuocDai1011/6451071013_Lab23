"""Sinh cac notebook LAB3 (moi bai mot file .ipynb).

Moi cell code co metadata {"cau": "<ten cau>"} de script tao bao cao doc lai.
Chay: python tools/build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MSSV = "6451071013"
SIGN = 'print("Họ tên: Mai Quốc Đại - MSSV: 6451071013 - Lớp: CQ.64.CNTT")'


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(cau, src, sign=True):
    src = src.strip("\n")
    if sign:
        src = SIGN + "\n" + src
    c = nbf.v4.new_code_cell(src)
    c.metadata["cau"] = cau
    return c


SUBS = {"K1": "4", "K2": "5", "K3": "6"}
try:  # nhan xet viet sau khi chay, luu trong tools/nhanxet.py
    from nhanxet import NHANXET
    SUBS.update(NHANXET)
except ImportError:
    pass


def save(name, title, cells):
    import sys
    if len(sys.argv) > 1 and name not in sys.argv[1:]:  # chi sinh lai cac bai duoc chi dinh
        return
    for c in cells:
        for k, v in SUBS.items():
            c.source = c.source.replace("{{" + k + "}}", v)
        if "pd.set_option" not in c.source:
            c.source = c.source.replace("import pandas as pd\n", 'import pandas as pd\npd.set_option("display.width", 140)\n', 1)
        c.source = c.source.replace(
            "keras.utils.set_random_seed(42)",
            'import tensorflow as tf\ntf.get_logger().setLevel("ERROR")  # ẩn cảnh báo không có GPU\nkeras.utils.set_random_seed(42)')
    cells = [c for c in cells if "{{" not in c.source]  # bo nhan xet chua viet
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["lab3_title"] = title
    nb.cells = [md(f"# {title}\n\n**Họ tên:** Mai Quốc Đại — **MSSV:** {MSSV} — **Lớp:** CQ.64.CNTT")] + cells
    path = ROOT / f"{MSSV}_LAB3_{name}.ipynb"
    nbf.write(nb, path)
    print("wrote", path.name)


# ---------------------------------------------------------------------------
# BAI 1 - Iris: DT, NB, KNN, NN
# ---------------------------------------------------------------------------
EVAL_HELPER = '''
def danh_gia(ten_mo_hinh, y_true, y_pred):
    """In Accuracy, classification report va ve confusion matrix bang heatmap."""
    print(f"===== {ten_mo_hinh} =====")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, target_names=iris.target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Nhãn dự đoán"); plt.ylabel("Nhãn thực tế")
    plt.title(f"Confusion matrix - {ten_mo_hinh}")
    plt.tight_layout(); plt.show()
'''

bai1 = [
    md("## Câu a) Đọc bộ dữ liệu Iris từ sklearn"),
    code("a", '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Input

keras.utils.set_random_seed(42)
pd.set_option("display.width", 120)
iris = load_iris()
print("Đã đọc bộ dữ liệu Iris:", type(iris))
'''),
    md("## Câu b) Khảo sát bộ dữ liệu Iris"),
    code("b", '''
print("Tên các thuộc tính:", iris.feature_names)
print("Thuộc tính phân lớp (các lớp):", [str(n) for n in iris.target_names])
print("Kích thước data:", iris.data.shape, "- kích thước target:", iris.target.shape)
print("Số lượng mẫu:", iris.data.shape[0], "- Số lượng thuộc tính:", iris.data.shape[1])
print("\\n5 mẫu dữ liệu đầu tiên:")
for i in range(5):
    print(iris.data[i], "->", iris.target[i], f"({iris.target_names[iris.target[i]]})")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("\\nThống kê mô tả:")
print(df.describe())
'''),
    md("## Câu c) Chuẩn hóa dữ liệu về đoạn [0,1]\nChia mỗi giá trị cho giá trị lớn nhất của cột tương ứng."),
    code("c", '''
X = iris.data
Y = iris.target
X_max = X.max(axis=0)
print("Giá trị lớn nhất mỗi cột:", X_max)
X = X / X_max
print("5 dòng đầu sau chuẩn hóa:\\n", np.round(X[:5], 4))
print("Min:", X.min(axis=0), "Max:", X.max(axis=0))
'''),
    md("## Câu d) Chia tập train/test theo tỷ lệ 8:2"),
    code("d", '''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
'''),
    md("## Câu e) Cây quyết định (Decision Tree)"),
    code("e", EVAL_HELPER + '''
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
danh_gia("Decision Tree", y_test, y_pred_dt)
'''),
    md("## Câu f) Naïve Bayes (GaussianNB)"),
    code("f", '''
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
danh_gia("Naive Bayes", y_test, y_pred_nb)
'''),
    md("## Câu g) K Nearest Neighbors"),
    code("g", '''
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
danh_gia("KNN (k=5)", y_test, y_pred_knn)
'''),
    md("## Câu h) Neural Network\n### Xây dựng và biên dịch mô hình"),
    code("h - xây dựng & biên dịch", '''
nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(10, activation="relu"),
    Dense(20, activation="relu"),
    Dense(len(iris.target_names), activation="softmax"),
])
nn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
nn.summary()
'''),
    md("### Huấn luyện (epochs=100, batch_size=1, verbose=1)"),
    code("h - huấn luyện", '''
history = nn.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)
'''),
    md("### Dự đoán và đánh giá"),
    code("h - dự đoán & đánh giá", '''
y_prob_nn = nn.predict(X_test, verbose=0)
y_pred_nn = np.argmax(y_prob_nn, axis=1)
print("Xác suất 3 mẫu test đầu:\\n", np.round(y_prob_nn[:3], 4))
danh_gia("Neural Network", y_test, y_pred_nn)
'''),
    md("## Câu i) Dự đoán 3 mẫu dữ liệu mới\nMẫu mới phải được chuẩn hóa bằng cùng giá trị max của tập dữ liệu gốc."),
    code("i", '''
X_new = np.array([[6.2, 2.9, 4.3, 1.3],
                  [5.1, 3.5, 1.4, 0.2],
                  [7.3, 2.8, 6.4, 2.1]])
X_new_norm = X_new / X_max
ket_qua = {
    "Decision Tree": dt.predict(X_new_norm),
    "Naive Bayes": nb.predict(X_new_norm),
    "KNN": knn.predict(X_new_norm),
    "Neural Network": np.argmax(nn.predict(X_new_norm, verbose=0), axis=1),
}
bang = pd.DataFrame({ten: iris.target_names[p] for ten, p in ket_qua.items()},
                    index=[str(x) for x in X_new.tolist()])
print(bang)
'''),
    md("## Câu j) Nhận xét, đánh giá hiệu suất các mô hình"),
    code("j", '''
from sklearn.metrics import precision_recall_fscore_support
rows = []
for ten, yp in [("Decision Tree", y_pred_dt), ("Naive Bayes", y_pred_nb),
                ("KNN", y_pred_knn), ("Neural Network", y_pred_nn)]:
    p, r, f, _ = precision_recall_fscore_support(y_test, yp, average="macro", zero_division=0)
    rows.append([ten, accuracy_score(y_test, yp), p, r, f])
so_sanh = pd.DataFrame(rows, columns=["Mô hình", "Accuracy", "Precision", "Recall", "F1"]).round(4)
print(so_sanh.to_string(index=False))
so_sanh.set_index("Mô hình").plot(kind="bar", figsize=(8, 4), ylim=(0.8, 1.02), rot=0)
plt.title("So sánh hiệu suất các mô hình trên tập test"); plt.tight_layout(); plt.show()

# Tập test chỉ có 30 mẫu -> bổ sung 10-fold cross validation để so sánh ổn định hơn
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("\\nAccuracy 10-fold cross validation trên toàn bộ 150 mẫu:")
for ten, m in [("Decision Tree", DecisionTreeClassifier(random_state=42)),
               ("Naive Bayes", GaussianNB()), ("KNN", KNeighborsClassifier(n_neighbors=5))]:
    s = cross_val_score(m, X, Y, cv=skf)
    print(f"  {ten:14s}: mean = {s.mean():.4f}, std = {s.std():.4f}")
'''),
    md("{{NHANXET_BAI1}}"),
]
save("Bai1", "Bài 1 (Trang 1) – Phân lớp Iris với Decision Tree, Naïve Bayes, KNN, Neural Network", bai1)

# ---------------------------------------------------------------------------
# BAI 2 - MNIST train/val/test
# ---------------------------------------------------------------------------
bai2 = [
    md("## Câu a) Đọc bộ dữ liệu MNIST từ Keras"),
    code("a", '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

keras.utils.set_random_seed(42)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Đã đọc MNIST")
'''),
    md("## Câu b) Kích thước tập train, test"),
    code("b", '''
print("train_images:", train_images.shape, "train_labels:", train_labels.shape)
print("test_images:", test_images.shape, "test_labels:", test_labels.shape)
'''),
    md("## Câu c) Chuẩn hóa pixel về [0,1]"),
    code("c", '''
train_images = train_images / 255.0
test_images = test_images / 255.0
print("Min:", train_images.min(), "Max:", train_images.max())
'''),
    md("## Câu d) Chia train/validation theo tỷ lệ 8:2"),
    code("d", '''
train_images, validation_images, train_labels, validation_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)
print("Train:", train_images.shape, train_labels.shape)
print("Validation:", validation_images.shape, validation_labels.shape)
'''),
    md("## Câu e) Xây dựng Neural Network"),
    code("e", '''
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax"),
])
model.summary()
'''),
    md("## Câu f) Biên dịch mô hình"),
    code("f", '''
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Đã biên dịch: optimizer=adam, loss=sparse_categorical_crossentropy, metrics=accuracy")
'''),
    md("## Câu g) Huấn luyện (epochs=5, batch_size=32)"),
    code("g", '''
history = model.fit(train_images, train_labels, epochs=5, batch_size=32,
                    validation_data=(validation_images, validation_labels), verbose=2)
'''),
    md("## Câu h) Biểu đồ accuracy, loss sau mỗi epoch"),
    code("h", '''
ep = range(1, len(history.history["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, history.history["accuracy"], "o-", label="train accuracy")
ax[0].plot(ep, history.history["val_accuracy"], "o-", label="validation accuracy")
ax[0].set_title("Accuracy theo epoch"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(ep, history.history["loss"], "o-", label="train loss")
ax[1].plot(ep, history.history["val_loss"], "o-", label="validation loss")
ax[1].set_title("Loss theo epoch"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
'''),
    md("## Câu i) Dự đoán phân lớp cho tập test"),
    code("i", '''
y_prob = model.predict(test_images, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
print("10 nhãn dự đoán đầu:", y_pred[:10])
print("10 nhãn thực tế đầu:", test_labels[:10])
'''),
    md("## Câu j) Accuracy trên tập test"),
    code("j", '''
print("Accuracy trên tập test:", round(accuracy_score(test_labels, y_pred), 4))
'''),
    md("## Câu k) Precision, Recall, F1-score"),
    code("k", '''
print(classification_report(test_labels, y_pred, digits=4))
'''),
    md("## Câu l) Confusion matrix (heatmap)"),
    code("l", '''
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Nhãn dự đoán"); plt.ylabel("Nhãn thực tế"); plt.title("Confusion matrix - MNIST")
plt.tight_layout(); plt.show()
'''),
    md("## Câu m) Kết quả phân lớp 5 ảnh đầu tiên của tập test"),
    code("m", '''
plt.figure(figsize=(12, 3))
for image_index in range(5):
    img = np.expand_dims(test_images[image_index], axis=0)
    pred = np.argmax(model.predict(img, verbose=0), axis=1)[0]
    print(f"Ảnh {image_index}: dự đoán = {pred}, thực tế = {test_labels[image_index]}")
    plt.subplot(1, 5, image_index + 1)
    plt.imshow(test_images[image_index], cmap="gray")
    plt.title(f"Dự đoán: {pred}\\nThực tế: {test_labels[image_index]}")
    plt.axis("off")
plt.tight_layout(); plt.show()
'''),
    md("## Câu n) 5 kết quả phân lớp sai đầu tiên"),
    code("n", '''
sai = np.where(y_pred != test_labels)[0]
print("Tổng số ảnh phân lớp sai:", len(sai))
plt.figure(figsize=(12, 3))
for k, idx in enumerate(sai[:5]):
    print(f"Ảnh index {idx}: dự đoán = {y_pred[idx]}, thực tế = {test_labels[idx]}")
    plt.subplot(1, 5, k + 1)
    plt.imshow(test_images[idx], cmap="gray")
    plt.title(f"#{idx} Dự đoán: {y_pred[idx]}\\nThực tế: {test_labels[idx]}", color="red")
    plt.axis("off")
plt.tight_layout(); plt.show()
'''),
]
save("Bai2", "Bài 2 (Trang 5) – Neural Network trên MNIST (chia train – test – validation)", bai2)

# ---------------------------------------------------------------------------
# BAI 3 - MNIST k-fold
# ---------------------------------------------------------------------------
bai3 = [
    md("## Câu a) Đọc bộ dữ liệu MNIST từ Keras"),
    code("a", '''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

keras.utils.set_random_seed(42)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Đã đọc MNIST")
'''),
    md("## Câu b) Kích thước tập train, test"),
    code("b", '''
print("train_images:", train_images.shape, "train_labels:", train_labels.shape)
print("test_images:", test_images.shape, "test_labels:", test_labels.shape)
'''),
    md("## Câu c) Chuẩn hóa pixel về [0,1]"),
    code("c", '''
train_images = train_images / 255.0
test_images = test_images / 255.0
print("Min:", train_images.min(), "Max:", train_images.max())
'''),
    md("## Câu d) Hàm create_model"),
    code("d", '''
def create_model():
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

create_model().summary()
'''),
    md("## Câu e) Huấn luyện bằng k-fold cross validation (k=10)\nMỗi fold: train 5 epoch, batch_size 32; lưu model vào mảng `models` và file `models/mnist_fold_<i>.keras`."),
    code("e", '''
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
models, accuracy_per_fold, loss_per_fold = [], [], []
os.makedirs("models", exist_ok=True)

for fold, (train_index, val_index) in enumerate(kfold.split(train_images), start=1):
    model = create_model()
    X_train_fold, X_val_fold = train_images[train_index], train_images[val_index]
    y_train_fold, y_val_fold = train_labels[train_index], train_labels[val_index]
    model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32,
              validation_data=(X_val_fold, y_val_fold), verbose=0)
    models.append(model)
    model.save(f"models/mnist_fold_{fold}.keras")
    loss, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_per_fold.append(acc); loss_per_fold.append(loss)
    print(f"Fold {fold:2d}: loss = {loss:.4f} - accuracy = {acc:.4f}")
'''),
    md("## Câu f) Mean / Std của Accuracy và Loss"),
    code("f", '''
print(f"Mean Accuracy: {np.mean(accuracy_per_fold):.4f}")
print(f"Std Accuracy : {np.std(accuracy_per_fold):.4f}")
print(f"Mean Loss    : {np.mean(loss_per_fold):.4f}")
print(f"Std Loss     : {np.std(loss_per_fold):.4f}")
'''),
    md("## Câu g) Biểu đồ accuracy, loss sau mỗi fold"),
    code("g", '''
folds = range(1, 11)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(folds, accuracy_per_fold, "o-"); ax[0].set_title("Accuracy theo fold")
ax[0].set_xlabel("Fold"); ax[0].set_xticks(list(folds))
ax[1].plot(folds, loss_per_fold, "o-", color="tab:orange"); ax[1].set_title("Loss theo fold")
ax[1].set_xlabel("Fold"); ax[1].set_xticks(list(folds))
plt.tight_layout(); plt.show()
'''),
    md("## Câu h) Dùng mô hình có Accuracy tốt nhất phân lớp tập test"),
    code("h", '''
index = int(np.argmax(accuracy_per_fold))
best_model = models[index]
print(f"Fold tốt nhất: {index + 1} (accuracy validation = {accuracy_per_fold[index]:.4f})")
y_pred = np.argmax(best_model.predict(test_images, verbose=0), axis=1)
print("10 nhãn dự đoán đầu:", y_pred[:10])
'''),
    md("## Câu i) Accuracy trên tập test"),
    code("i", '''
print("Accuracy trên tập test:", round(accuracy_score(test_labels, y_pred), 4))
'''),
    md("## Câu j) Precision, Recall, F1-score"),
    code("j", '''
print(classification_report(test_labels, y_pred, digits=4))
'''),
    md("## Câu k) Confusion matrix (heatmap)"),
    code("k", '''
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Nhãn dự đoán"); plt.ylabel("Nhãn thực tế"); plt.title("Confusion matrix - mô hình fold tốt nhất")
plt.tight_layout(); plt.show()
'''),
    md("## Câu l) Kết quả phân lớp 5 ảnh đầu tiên của tập test"),
    code("l", '''
plt.figure(figsize=(12, 3))
for image_index in range(5):
    img = np.expand_dims(test_images[image_index], axis=0)
    pred = np.argmax(best_model.predict(img, verbose=0), axis=1)[0]
    print(f"Ảnh {image_index}: dự đoán = {pred}, thực tế = {test_labels[image_index]}")
    plt.subplot(1, 5, image_index + 1)
    plt.imshow(test_images[image_index], cmap="gray")
    plt.title(f"Dự đoán: {pred}\\nThực tế: {test_labels[image_index]}")
    plt.axis("off")
plt.tight_layout(); plt.show()
'''),
]
save("Bai3", "Bài 3 (Trang 7) – Neural Network trên MNIST (k-fold cross validation)", bai3)

# ---------------------------------------------------------------------------
# BAI 4 - Iris k-fold
# ---------------------------------------------------------------------------
bai4 = [
    md("## Câu a) Đọc bộ dữ liệu Iris"),
    code("a", '''
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

keras.utils.set_random_seed(42)
iris = load_iris()
X, Y = iris.data, iris.target
print("X:", X.shape, "Y:", Y.shape, "- Các lớp:", [str(n) for n in iris.target_names])
'''),
    md("## Câu b) Chuẩn hóa dữ liệu về [0,1]"),
    code("b", '''
X_max = X.max(axis=0)
X = X / X_max
print("Max mỗi cột ban đầu:", X_max)
print("Min/Max sau chuẩn hóa:", X.min(axis=0), X.max(axis=0))
'''),
    md("## Câu c) Hàm create_model (giống câu 1.h)"),
    code("c", '''
def create_model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(10, activation="relu"),
        Dense(20, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

create_model().summary()
'''),
    md("## Câu d) Huấn luyện bằng k-fold cross validation (k=10)\nMỗi fold huấn luyện 100 epoch, batch_size=1 (giống câu 1.h). Precision/Recall/F1 lấy trung bình macro."),
    code("d", '''
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
models = []
acc_list, pre_list, rec_list, f1_list = [], [], [], []
pre_cls, rec_cls, f1_cls = [], [], []

for fold, (train_index, val_index) in enumerate(kfold.split(X), start=1):
    model = create_model()
    model.fit(X[train_index], Y[train_index], epochs=100, batch_size=1, verbose=0)
    models.append(model)
    y_pred = np.argmax(model.predict(X[val_index], verbose=0), axis=1)
    y_true = Y[val_index]
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pc, rc, fc, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    acc_list.append(acc); pre_list.append(p); rec_list.append(r); f1_list.append(f)
    pre_cls.append(pc); rec_cls.append(rc); f1_cls.append(fc)
    print(f"Fold {fold:2d}: accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")
'''),
    md("## Câu e) Giá trị trung bình sau k-fold"),
    code("e", '''
print(f"Average Accuracy : {np.mean(acc_list):.4f}")
print(f"Average Recall   : {np.mean(rec_list):.4f}")
print(f"Average Precision: {np.mean(pre_list):.4f}")
print(f"Average F1-score : {np.mean(f1_list):.4f}")
'''),
    md("## Câu f) Giá trị trung bình cho từng class"),
    code("f", '''
bang = pd.DataFrame({
    "Average Precision": np.mean(pre_cls, axis=0),
    "Average Recall": np.mean(rec_cls, axis=0),
    "Average F1-score": np.mean(f1_cls, axis=0),
}, index=iris.target_names).round(4)
print(bang)
'''),
    md("## Câu g) Dùng mô hình có F1-score tốt nhất dự đoán 3 mẫu mới"),
    code("g", '''
best = int(np.argmax(f1_list))
print(f"Mô hình tốt nhất: fold {best + 1} (F1 = {f1_list[best]:.4f})")
X_new = np.array([[6.2, 2.9, 4.3, 1.3],
                  [5.1, 3.5, 1.4, 0.2],
                  [7.3, 2.8, 6.4, 2.1]])
prob = models[best].predict(X_new / X_max, verbose=0)
np.set_printoptions(suppress=True)
for x, p in zip(X_new.tolist(), prob):
    print(f"{x} -> {iris.target_names[np.argmax(p)]}  (xác suất: {np.round(p, 4)})")
'''),
]
save("Bai4", "Bài 4 (Trang 9) – Neural Network trên Iris (k-fold cross validation)", bai4)

# ---------------------------------------------------------------------------
# BAI 5 (Phan 1) - Game sales regression
# ---------------------------------------------------------------------------
bai5 = [
    md("## Câu a) Đọc dữ liệu train và test"),
    code("a", '''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, LambdaCallback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

keras.utils.set_random_seed(42)
train_df = pd.read_csv("data/sales_data_training.csv")
test_df = pd.read_csv("data/sales_data_test.csv")
print("Train:", train_df.shape, "- Test:", test_df.shape)
print(train_df.head())
'''),
    md("## Câu b) Tách thuộc tính cần dự đoán total_earnings"),
    code("b", '''
X_train_raw = train_df.drop(columns=["total_earnings"]).values
y_train_raw = train_df[["total_earnings"]].values
X_test_raw = test_df.drop(columns=["total_earnings"]).values
y_test_raw = test_df[["total_earnings"]].values
print("Các thuộc tính đầu vào:", list(train_df.drop(columns=["total_earnings"]).columns))
print("X_train:", X_train_raw.shape, "y_train:", y_train_raw.shape)
print("X_test:", X_test_raw.shape, "y_test:", y_test_raw.shape)
'''),
    md("## Câu c) Chuyển dữ liệu về [0,1]\nScaler được fit trên tập train và áp dụng cho tập test (tránh rò rỉ dữ liệu)."),
    code("c", '''
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_all = x_scaler.fit_transform(X_train_raw)
y_train_all = y_scaler.fit_transform(y_train_raw)
X_test = x_scaler.transform(X_test_raw)
y_test = y_scaler.transform(y_test_raw)
print("X_train min/max:", X_train_all.min(), X_train_all.max())
print("y_train min/max:", y_train_all.min(), y_train_all.max())
print("y_test  min/max:", round(y_test.min(), 4), round(y_test.max(), 4))
'''),
    md("## Câu d) Chia train/validation tỷ lệ 8:2"),
    code("d", '''
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "- Validation:", X_val.shape)
'''),
    md("## Câu e) Xây dựng và biên dịch mô hình"),
    code("e", '''
model = Sequential([
    Input(shape=(9,)),
    Dense(50, activation="relu"),
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="mean_squared_error", metrics=["mean_absolute_error"])
model.summary()
'''),
    md("## Câu f, g) Huấn luyện 100 epoch, in train_loss/val_loss mỗi epoch và lưu model tốt nhất (val_loss thấp nhất)"),
    code("f, g", '''
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint("models/best_sales_model.keras", monitor="val_loss",
                             save_best_only=True, mode="min", verbose=0)
in_loss = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
    f"Epoch {epoch + 1:3d}: train_loss = {logs['loss']:.6f} - val_loss = {logs['val_loss']:.6f}"))
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint, in_loss], verbose=0)
best_epoch = int(np.argmin(history.history["val_loss"])) + 1
print(f"\\nModel tốt nhất ở epoch {best_epoch}, val_loss = {min(history.history['val_loss']):.6f}")
print("Đã lưu vào models/best_sales_model.keras")

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE (thang log)"); plt.legend()
plt.title("Train loss và Validation loss"); plt.tight_layout(); plt.show()
'''),
    md("## Câu h) Đánh giá Loss và Accuracy trên tập test\nĐây là bài toán hồi quy nên \"accuracy\" được thể hiện qua MAE và hệ số xác định R² (mức độ mô hình giải thích được biến thiên của doanh thu)."),
    code("h", '''
best_model = load_model("models/best_sales_model.keras")
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss (MSE, thang [0,1]): {test_loss:.6f}")
print(f"Test MAE (thang [0,1])     : {test_mae:.6f}")

y_pred = y_scaler.inverse_transform(best_model.predict(X_test, verbose=0))
print(f"MAE theo đơn vị gốc ($)  : {mean_absolute_error(y_test_raw, y_pred):,.2f}")
print(f"RMSE theo đơn vị gốc ($) : {np.sqrt(mean_squared_error(y_test_raw, y_pred)):,.2f}")
print(f"R² (accuracy hồi quy)     : {r2_score(y_test_raw, y_pred):.4f}")

so_sanh = pd.DataFrame({"Thực tế": y_test_raw.ravel(), "Dự đoán": y_pred.ravel().round(0)}).head(5)
print("\\n5 mẫu test đầu tiên:\\n", so_sanh)

plt.figure(figsize=(5, 5))
plt.scatter(y_test_raw, y_pred, s=15, alpha=0.6)
lim = [y_test_raw.min(), y_test_raw.max()]
plt.plot(lim, lim, "r--"); plt.xlabel("Doanh thu thực tế"); plt.ylabel("Doanh thu dự đoán")
plt.title("Dự đoán vs Thực tế (tập test)"); plt.tight_layout(); plt.show()
'''),
]
save("Bai5", "Bài 5 (Trang 9) – Neural Network dự đoán doanh thu kinh doanh game", bai5)

# ---------------------------------------------------------------------------
# PHAN 2 - Bai 2 (Trang 18-19): Mall customers
# ---------------------------------------------------------------------------
CLUSTER_HELPER = '''
def elbow(X, title):
    inertia = []
    for n in range(1, 11):
        algorithm = KMeans(n_clusters=n, init="k-means++", n_init=10, random_state=42)
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, 11), inertia, "o")
    plt.plot(np.arange(1, 11), inertia, "-", alpha=0.5)
    plt.xlabel("Number of Clusters"); plt.ylabel("Inertia"); plt.title(title)
    plt.tight_layout(); plt.show()
    for n, v in zip(range(1, 11), inertia):
        print(f"k = {n:2d}: inertia = {v:,.1f}")
    return inertia
'''

mall = [
    md("## Câu 1) Nhập dữ liệu, thống kê cơ bản, xử lý dữ liệu thiếu"),
    code("1", '''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.renderers.default = "notebook_connected"

data = pd.read_csv("data/Mall_Customers.csv")
print("Kích thước:", data.shape)
print(data.head())
print("\\nKiểu dữ liệu:"); print(data.dtypes)
print("\\nThống kê mô tả:"); print(data.describe())
print("\\nSố giá trị thiếu mỗi cột:"); print(data.isnull().sum())
# Xử lý dữ liệu thiếu: điền trung vị cho cột số, giá trị phổ biến nhất cho cột phân loại
for col in data.columns:
    if data[col].isnull().any():
        is_num = pd.api.types.is_numeric_dtype(data[col])
        data[col] = data[col].fillna(data[col].median() if is_num else data[col].mode()[0])
print("\\nSau xử lý, tổng số giá trị thiếu:", int(data.isnull().sum().sum()))
'''),
    md("## Câu 2) Histogram phân phối Tuổi, Thu nhập, Điểm mua sắm"),
    code("2", '''
plt.style.use("fivethirtyeight")
plt.figure(1, figsize=(15, 6))
n = 0
for x in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(data[x], bins=20, kde=True, stat="density")
    plt.title("Distplot of {}".format(x))
plt.show()
'''),
    md("## Câu 3) Thống kê số mẫu theo Giới tính"),
    code("3", '''
print(data["Gender"].value_counts())
plt.figure(1, figsize=(15, 5))
sns.countplot(y="Gender", data=data, hue="Gender")
plt.show()
'''),
    md("## Câu 4) Scatter plot Tuổi – Thu nhập theo Giới tính"),
    code("4", '''
plt.figure(1, figsize=(15, 6))
for gender in ["Male", "Female"]:
    plt.scatter(x="Age", y="Annual Income (k$)", data=data[data["Gender"] == gender],
                s=200, alpha=0.5, label=gender)
plt.xlabel("Age"), plt.ylabel("Annual Income (k$)")
plt.title("Age vs Annual Income w.r.t Gender")
plt.legend(); plt.show()
'''),
    md("## Câu 5) Scatter plot Thu nhập – Điểm mua sắm theo Giới tính"),
    code("5", '''
plt.figure(1, figsize=(15, 6))
for gender in ["Male", "Female"]:
    plt.scatter(x="Annual Income (k$)", y="Spending Score (1-100)", data=data[data["Gender"] == gender],
                s=200, alpha=0.5, label=gender)
plt.xlabel("Annual Income (k$)"), plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score w.r.t Gender")
plt.legend(); plt.show()
'''),
    md("## Câu 6) Gom cụm Tuổi – Điểm mua sắm: Elbow + k-Means"),
    code("6 - elbow", CLUSTER_HELPER + '''
X1 = data[["Age", "Spending Score (1-100)"]].iloc[:, :].values
inertia1 = elbow(X1, "Elbow - Age & Spending Score")
'''),
    md("{{ELBOW1}}"),
    code("6 - k-Means", '''
algorithm = KMeans(n_clusters={{K1}}, init="k-means++", n_init=10, random_state=42)
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
print("Số phần tử mỗi cụm:", np.bincount(labels1))
print("Tâm cụm (Age, Spending Score):\\n", np.round(centroids1, 2))
'''),
    md("## Câu 7) Biểu diễn kết quả gom cụm"),
    code("7", '''
plt.figure(1, figsize=(15, 7))
plt.clf()
plt.scatter(x="Age", y="Spending Score (1-100)", data=data, c=labels1, s=200)
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=300, c="red", alpha=0.5)
plt.ylabel("Spending Score (1-100)"), plt.xlabel("Age")
plt.title("Phân cụm theo Age & Spending Score (k={{K1}})")
plt.show()
'''),
    md("## Câu 8) Đánh giá bằng Silhouette coefficient"),
    code("8", '''
print("Silhouette (Age, Spending Score):", round(metrics.silhouette_score(X1, labels1, metric="euclidean"), 4))
'''),
    md("## Câu 9.1) Gom cụm Thu nhập – Điểm mua sắm"),
    code("9.1 - elbow", '''
X2 = data[["Annual Income (k$)", "Spending Score (1-100)"]].iloc[:, :].values
inertia2 = elbow(X2, "Elbow - Annual Income & Spending Score")
'''),
    md("{{ELBOW2}}"),
    code("9.1 - k-Means, biểu đồ, silhouette", '''
algorithm = KMeans(n_clusters={{K2}}, init="k-means++", n_init=10, random_state=42)
algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
print("Số phần tử mỗi cụm:", np.bincount(labels2))
print("Tâm cụm (Income, Spending Score):\\n", np.round(centroids2, 2))

plt.figure(1, figsize=(15, 7))
plt.clf()
plt.scatter(x="Annual Income (k$)", y="Spending Score (1-100)", data=data, c=labels2, s=200)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c="red", alpha=0.5)
plt.ylabel("Spending Score (1-100)"), plt.xlabel("Annual Income (k$)")
plt.title("Phân cụm theo Annual Income & Spending Score (k={{K2}})")
plt.show()
print("Silhouette (Income, Spending Score):", round(metrics.silhouette_score(X2, labels2, metric="euclidean"), 4))
'''),
    md("## Câu 9.2) Gom cụm Tuổi – Thu nhập – Điểm mua sắm (3D)"),
    code("9.2 - elbow", '''
X3 = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].iloc[:, :].values
inertia3 = elbow(X3, "Elbow - Age, Annual Income & Spending Score")
'''),
    md("{{ELBOW3}}"),
    code("9.2 - k-Means & silhouette", '''
algorithm = KMeans(n_clusters={{K3}}, init="k-means++", n_init=10, random_state=42)
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_
print("Số phần tử mỗi cụm:", np.bincount(labels3))
print("Tâm cụm (Age, Income, Spending Score):\\n", np.round(centroids3, 2))
print("Silhouette (Age, Income, Spending Score):", round(metrics.silhouette_score(X3, labels3, metric="euclidean"), 4))
'''),
    code("9.2 - biểu đồ 3D (plotly)", '''
data["label3"] = labels3
trace1 = go.Scatter3d(
    x=data["Age"], y=data["Spending Score (1-100)"], z=data["Annual Income (k$)"],
    mode="markers",
    marker=dict(color=data["label3"], size=10, line=dict(color=data["label3"], width=12), opacity=0.8),
)
layout = go.Layout(title="Clusters", scene=dict(
    xaxis=dict(title="Age"), yaxis=dict(title="Spending Score"), zaxis=dict(title="Annual Income")))
fig = go.Figure(data=[trace1], layout=layout)
fig.show()
'''),
    md("Biểu đồ plotly ở trên có thể xoay tương tác khi mở notebook. Bản tĩnh bằng matplotlib để chèn vào báo cáo:"),
    code("9.2 - biểu đồ 3D (matplotlib)", '''
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(data["Age"], data["Spending Score (1-100)"], data["Annual Income (k$)"],
                c=labels3, cmap="plasma", s=60, alpha=0.8)
ax.scatter(centroids3[:, 0], centroids3[:, 2], centroids3[:, 1], c="red", s=300, marker="X")
ax.set_xlabel("Age"); ax.set_ylabel("Spending Score"); ax.set_zlabel("Annual Income")
ax.set_title("Clusters (k={{K3}})")
plt.tight_layout(); plt.show()
'''),
    md("{{NHANXET_MALL}}"),
]
save("Bai2_Phan2", "Bài 2 (Trang 18 – 19) – Gom cụm dữ liệu Mall Customers bằng k-Means", mall)

# ---------------------------------------------------------------------------
# PHAN 2 - Bai 5 (Trang 26): MNIST k-Means
# ---------------------------------------------------------------------------
mnist_km = [
    md("## Đọc và chuẩn hóa dữ liệu MNIST\nMỗi ảnh 28x28 được làm phẳng thành vector 784 chiều, giá trị pixel chia 255."),
    code("Đọc dữ liệu", '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             adjusted_rand_score, normalized_mutual_info_score)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = train_images.reshape(len(train_images), -1) / 255.0
X_test = test_images.reshape(len(test_images), -1) / 255.0
print("X_train:", X_train.shape, "- X_test:", X_test.shape)
'''),
    md("## Gom cụm bằng k-Means (k = 10)"),
    code("Gom cụm k-Means", '''
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(X_train)
clusters_train = kmeans.labels_
print("Số phần tử mỗi cụm:", np.bincount(clusters_train))
print("Inertia:", round(kmeans.inertia_, 2))
'''),
    md("## Tâm các cụm\nMỗi tâm cụm là một vector 784 chiều, hiển thị lại dưới dạng ảnh 28x28."),
    code("Hiển thị tâm cụm", '''
fig, ax = plt.subplots(2, 5, figsize=(10, 4.5))
for i, axi in enumerate(ax.flat):
    axi.imshow(kmeans.cluster_centers_[i].reshape(28, 28), cmap="binary")
    axi.set_title(f"Cụm {i}"); axi.axis("off")
plt.suptitle("Tâm 10 cụm k-Means"); plt.tight_layout(); plt.show()
'''),
    md("## Gán nhãn cho cụm\nCụm không có nhãn sẵn, nên mỗi cụm được gán chữ số xuất hiện nhiều nhất trong cụm đó (trên tập train)."),
    code("Ánh xạ cụm → nhãn", '''
cluster_to_label = {}
for c in range(10):
    nhan_trong_cum = train_labels[clusters_train == c]
    cluster_to_label[c] = int(np.bincount(nhan_trong_cum).argmax())
    ty_le = np.mean(nhan_trong_cum == cluster_to_label[c])
    print(f"Cụm {c}: {len(nhan_trong_cum):5d} ảnh -> nhãn {cluster_to_label[c]} (độ thuần khiết {ty_le:.2%})")
print("Các chữ số không được cụm nào đại diện:", sorted(set(range(10)) - set(cluster_to_label.values())))
'''),
    md("## So sánh kết quả gom cụm với nhãn có sẵn"),
    code("Đánh giá trên tập train và test", '''
y_train_pred = np.array([cluster_to_label[c] for c in clusters_train])
clusters_test = kmeans.predict(X_test)
y_test_pred = np.array([cluster_to_label[c] for c in clusters_test])
print("Accuracy (train):", round(accuracy_score(train_labels, y_train_pred), 4))
print("Accuracy (test) :", round(accuracy_score(test_labels, y_test_pred), 4))
print("Adjusted Rand Index (test):", round(adjusted_rand_score(test_labels, clusters_test), 4))
print("NMI (test):", round(normalized_mutual_info_score(test_labels, clusters_test), 4))
print("\\nClassification report (test):")
print(classification_report(test_labels, y_test_pred, digits=4, zero_division=0))
'''),
    code("Confusion matrix", '''
cm = confusion_matrix(test_labels, y_test_pred, labels=range(10))
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Nhãn suy ra từ cụm"); plt.ylabel("Nhãn thực tế")
plt.title("Confusion matrix - k-Means trên MNIST (test)")
plt.tight_layout(); plt.show()
'''),
    code("Bảng chéo cụm × nhãn", '''
bang = pd.crosstab(pd.Series(clusters_test, name="Cụm"), pd.Series(test_labels, name="Nhãn thực tế"))
print(bang)
'''),
    md("{{NHANXET_MNIST_KM}}"),
]
save("Bai5_Phan2", "Bài 5 (Trang 26) – Gom cụm MNIST bằng k-Means và so sánh với nhãn có sẵn", mnist_km)

# ---------------------------------------------------------------------------
# PHAN 2 - Bai 6 (Trang 26): image color quantization
# ---------------------------------------------------------------------------
img_km = [
    md("## Khai báo thư viện"),
    code("Khai báo thư viện", '''
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
print("Đã import thư viện")
'''),
    md("## Đọc ảnh và hiển thị"),
    code("Đọc ảnh", '''
image = Image.open("data/image.png").convert("RGB")
axes = plt.axes(xticks=[], yticks=[])
axes.imshow(image)
plt.show()
'''),
    md("## Chuyển ảnh thành mảng số, lấy chiều cao/rộng"),
    code("Chuyển thành mảng", '''
imarr = np.array(image)
rows = imarr.shape[0]
cols = imarr.shape[1]
print("Kích thước ảnh:", imarr.shape)
print("Số màu khác nhau trong ảnh gốc:", len(np.unique(imarr.reshape(-1, 3), axis=0)))
'''),
    md("## Đưa giá trị màu về [0,1] và chuyển mảng 3 chiều thành 2 chiều"),
    code("Chuẩn hóa & reshape", '''
data = imarr / 255.0
data = data.reshape(rows * cols, 3)
print(data.shape)
'''),
    md("## Gom cụm với 32 màu và lấy mảng ảnh mới"),
    code("k-Means 32 màu", '''
kmeans = KMeans(n_clusters=32, max_iter=300, n_init=1, random_state=42)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
print("Bảng màu mới:", kmeans.cluster_centers_.shape)
'''),
    md("## Khôi phục mảng 3 chiều và so sánh với ảnh gốc"),
    code("So sánh ảnh", '''
img_recolored = new_colors.reshape(imarr.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(image)
ax[0].set_title("Original Image", size=16)
ax[1].imshow(img_recolored)
ax[1].set_title("32-color Image", size=16)
plt.show()
'''),
    md("## Lưu ảnh kết quả và so sánh kích thước file"),
    code("Lưu ảnh", '''
os.makedirs("output", exist_ok=True)
img_out = img_recolored * 255
new_img = Image.fromarray(img_out.astype(np.uint8))
new_img.save("output/compressed_img2.png", optimize=True)
# ảnh 32 màu có thể lưu dạng bảng màu (palette) 5 bit/pixel -> nhỏ hơn nhiều
new_img.quantize(colors=32, method=Image.Quantize.FASTOCTREE).save("output/compressed_img2_palette.png", optimize=True)
image.save("output/original.png", optimize=True)
for f in ["output/original.png", "output/compressed_img2.png", "output/compressed_img2_palette.png"]:
    print(f"{f:40s}: {os.path.getsize(f) / 1024:8.1f} KB")
'''),
    md("## Thử với các số màu khác nhau"),
    code("So sánh nhiều mức k", '''
fig, ax = plt.subplots(1, 4, figsize=(18, 4), subplot_kw=dict(xticks=[], yticks=[]))
for axi, k in zip(ax, [4, 8, 16, 64]):
    km = KMeans(n_clusters=k, n_init=1, random_state=42).fit(data)
    out = (km.cluster_centers_[km.labels_].reshape(imarr.shape) * 255).astype(np.uint8)
    path = f"output/compressed_{k}_colors.png"
    Image.fromarray(out).quantize(colors=k, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
    axi.imshow(out); axi.set_title(f"{k} màu - {os.path.getsize(path) / 1024:.0f} KB")
plt.tight_layout(); plt.show()
'''),
    md("{{NHANXET_IMG}}"),
]
save("Bai6_Phan2", "Bài 6 (Trang 26) – Giảm số màu ảnh bằng k-Means", img_km)

# ---------------------------------------------------------------------------
# Bai tap them 5 (Trang 29): Breast cancer
# ---------------------------------------------------------------------------
bc = [
    md("## Đọc dữ liệu Breast Cancer Wisconsin (Diagnostic)\nFile `wdbc.data`: cột 1 là ID, cột 2 là chẩn đoán (M = ác tính, B = lành tính), 30 cột còn lại là đặc trưng của nhân tế bào."),
    code("Đọc dữ liệu", '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dac_trung = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
             "concavity", "concave_points", "symmetry", "fractal_dimension"]
columns = ["id", "diagnosis"] + [f"{f}_{s}" for s in ["mean", "se", "worst"] for f in dac_trung]
df = pd.read_csv("data/breast_cancer/wdbc.data", header=None, names=columns)
print("Kích thước:", df.shape)
print(df.iloc[:5, :8])
print("\\nSố giá trị thiếu:", int(df.isnull().sum().sum()))
print("\\nPhân bố nhãn chẩn đoán:"); print(df["diagnosis"].value_counts())
'''),
    md("## Tiền xử lý\nCác đặc trưng có thang đo rất khác nhau (ví dụ `area` hàng nghìn, `smoothness` ~0.1), nên chuẩn hóa z-score trước khi gom cụm. Nhãn `diagnosis` chỉ dùng để đánh giá, không đưa vào thuật toán."),
    code("Tiền xử lý", '''
X = df.drop(columns=["id", "diagnosis"]).values
y_true = (df["diagnosis"] == "M").astype(int).values   # 1 = ác tính, 0 = lành tính
X_scaled = StandardScaler().fit_transform(X)
print("X_scaled:", X_scaled.shape, "- mean ~", round(X_scaled.mean(), 4), "- std ~", round(X_scaled.std(), 4))
'''),
    md("## Khảo sát số cụm: Elbow và Silhouette"),
    code("Elbow & Silhouette", '''
inertia, sil = [], []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_); sil.append(metrics.silhouette_score(X_scaled, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (Inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "o-", color="tab:green"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.show()
for k, i, s in zip(ks, inertia, sil):
    print(f"k = {k:2d}: inertia = {i:9.1f} - silhouette = {s:.4f}")
'''),
    md("## Gom cụm k-Means với k = 2 (lành tính / ác tính)"),
    code("k-Means k=2", '''
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
print("Số phần tử mỗi cụm:", np.bincount(clusters))
print("Silhouette:", round(metrics.silhouette_score(X_scaled, clusters), 4))
# Gán nhãn cho cụm theo đa số
cluster_to_label = {c: int(np.bincount(y_true[clusters == c]).argmax()) for c in range(2)}
for c, l in cluster_to_label.items():
    print(f"Cụm {c} -> {'Ác tính (M)' if l == 1 else 'Lành tính (B)'}")
y_pred = np.array([cluster_to_label[c] for c in clusters])
'''),
    md("## So sánh với nhãn chẩn đoán thực tế"),
    code("Đánh giá", '''
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Adjusted Rand Index:", round(metrics.adjusted_rand_score(y_true, clusters), 4))
print(classification_report(y_true, y_pred, target_names=["Lành tính (B)", "Ác tính (M)"], digits=4))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["B", "M"], yticklabels=["B", "M"])
plt.xlabel("Nhãn suy ra từ cụm"); plt.ylabel("Chẩn đoán thực tế"); plt.title("Confusion matrix")
plt.tight_layout(); plt.show()
'''),
    md("## Trực quan hóa bằng PCA 2 chiều"),
    code("PCA 2D", '''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
C_pca = pca.transform(kmeans.cluster_centers_)
print("Tỷ lệ phương sai giải thích bởi 2 thành phần:", np.round(pca.explained_variance_ratio_, 4))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="coolwarm", s=20, alpha=0.7)
ax[0].scatter(C_pca[:, 0], C_pca[:, 1], c="black", s=250, marker="X")
ax[0].set_title("Kết quả gom cụm k-Means")
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="coolwarm", s=20, alpha=0.7)
ax[1].set_title("Nhãn thực tế (đỏ = ác tính, xanh = lành tính)")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.show()
'''),
    md("{{NHANXET_BC}}"),
]
save("Bai5_BaiTapThem", "Bài 5 (Trang 29) – Gom cụm khối u lành tính / ác tính (Breast Cancer Wisconsin)", bc)
