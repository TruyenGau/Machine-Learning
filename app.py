import tkinter as tk
import seaborn as sns
from tkinter import ttk, filedialog
from tkinter.ttk import Scrollbar

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np

df = pd.DataFrame


def load_csv_variables():
    global variables
    global df
    file_path = filedialog.askopenfilename(filetypes=[("Tệp CSV", "*.csv")])
    if file_path:
        try:
            with open(file_path, 'r') as file:
                global df
                df = pd.read_csv(file_path)
                variables = list(df.columns)
                input_variable_listbox.delete(0, tk.END)
                input_variable_listbox.insert(tk.END, *variables)
                target_variable_combobox['values'] = list(df.columns)
                file_name = file_path.split('/')[-1]  # Lấy tên file từ đường dẫn
                file_label.config(text=file_name)  # Hiển thị tên file
        except Exception as e:
            status_label.config(text=f"Lỗi khi tải tệp CSV: {e}", fg="red")


def add_input_variable():
    selected_indices = input_variable_listbox.curselection()
    for idx in selected_indices[::-1]:
        variable = input_variable_listbox.get(idx)
        input_variable_listbox.delete(idx)
        selected_input_variables_listbox.insert(tk.END, variable)
        selected_input_variables.append(variable)


def remove_input_variable():
    selected_indices = selected_input_variables_listbox.curselection()
    for idx in selected_indices[::-1]:
        variable = selected_input_variables_listbox.get(idx)
        selected_input_variables_listbox.delete(idx)
        input_variable_listbox.insert(tk.END, variable)
        selected_input_variables.remove(variable)


def get_model():
    selected_value = model_combobox.get()
    return selected_value


def get_target():
    selected_value = target_variable_combobox.get()
    return selected_value


def plot_confusion_matrix(cm, classes, normalize=False, title='Ma trận nhầm lẫn', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Ma trận nhầm lẫn đã chuẩn hóa")
    else:
        print('Ma trận nhầm lẫn, chưa được chuẩn hóa')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Nhãn thực')
    ax.set_xlabel('Nhãn dự đoán')

    return fig


def show_new_page():
    model = get_model()
    if model == "Hồi quy Tuyến tính":
        plot_linear_regression()
    elif model == "Hồi quy Logistic":
        plot_logistic_regression()


import io


def show_info():
    # Xóa nội dung cũ trong Text
    info_text.delete('1.0', tk.END)

    # Số hàng và số cột
    num_rows, num_cols = df.shape
    info_text.insert(tk.END, f"Số hàng: {num_rows}\n")
    info_text.insert(tk.END, f"Số cột: {num_cols}\n\n")

    # Tên của các cột dạng chữ
    string_columns = df.select_dtypes(include='object').columns.tolist()
    info_text.insert(tk.END, "Các cột dạng chữ:\n")
    for col in string_columns:
        info_text.insert(tk.END, f"- {col}\n")

    # Tên của các cột dạng số
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    info_text.insert(tk.END, "\nCác cột dạng số:\n")
    for col in numeric_columns:
        info_text.insert(tk.END, f"- {col}\n")

    # Số lượng dữ liệu dạng chuỗi và số lượng kí tự dạng số
    num_string_data = df[string_columns].applymap(lambda x: isinstance(x, str)).sum().sum()
    num_numeric_chars = df[numeric_columns].applymap(lambda x: isinstance(x, (int, float))).sum().sum()
    info_text.insert(tk.END, f"\nSố lượng dữ liệu dạng chuỗi: {num_string_data}\n")
    info_text.insert(tk.END, f"Số lượng kí tự dạng số: {num_numeric_chars}\n")


def show_data():
    data_table.delete('1.0', tk.END)  # Xóa nội dung cũ trước khi hiển thị dữ liệu mới
    data_table.insert(tk.END, df.head(50).to_string(max_cols=20))  # Hiển thị 50 dòng đầu của dữ liệu, giới hạn độ dài của mỗi cột là 20 ký tự


def process_missing_data():
    # Xử lý dữ liệu thiếu: thêm dữ liệu vào các cột thiếu
    # Ví dụ: điền giá trị trung bình vào các cột số, và điền giá trị phổ biến nhất vào các cột chữ
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in numerical_columns:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

    for col in categorical_columns:
        most_common_value = df[col].mode()[0]
        df.fillna({col: most_common_value}, inplace=True)

    # Hiển thị thông tin trong data_table
    data_table.delete('1.0', tk.END)  # Xóa nội dung cũ trước khi hiển thị dữ liệu mới
    data_table.insert(tk.END, df.head().to_string(max_cols=20))  # Hiển thị 5 dòng đầu tiên của DataFrame


#biểu đồ thống kê của các biến
def show_statistics():
    # Lấy tên của cột cần thống kê
    selected_column = get_target()

    # Tạo cửa sổ mới
    new_window = tk.Toplevel(root)

    # Kiểm tra kiểu dữ liệu của cột
    if df[selected_column].dtype == 'object':
        # Vẽ biểu đồ thống kê cho biến dạng chữ
        plt.figure(figsize=(8, 6))
        sns.countplot(x=selected_column, data=df)
        plt.title(f"Biểu đồ thống kê của {selected_column}")
        plt.xticks(rotation=45)
        plt.tight_layout()

    else:
        # Vẽ biểu đồ thống kê cho biến dạng số
        plt.figure(figsize=(8, 6))
        sns.histplot(df[selected_column])
        plt.title(f"Phân phối của {selected_column}")
        plt.tight_layout()

    # Tạo FigureCanvasTkAgg và hiển thị trên cửa sổ mới
    canvas = FigureCanvasTkAgg(plt.gcf(), master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# hồi quy tuyến tính đơn biến
def plot_linear_regression():
    # Lấy tên biến phụ thuộc
    dependent_variable = get_target()

    # Lấy tên biến độc lập
    independent_variable = selected_input_variables_listbox.get(tk.ACTIVE)

    # Lấy dữ liệu
    X = df[[independent_variable]]
    y = df[dependent_variable]

    # Khởi tạo mô hình hồi quy tuyến tính
    model = LinearRegression()

    # Huấn luyện mô hình
    model.fit(X, y)

    # Dự đoán
    y_pred = model.predict(X)

    # Vẽ biểu đồ scatter plot và đường hồi quy
    plt.figure()
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    plt.title(f"Linear Regression: {dependent_variable} vs {independent_variable}")
    plt.grid(True)

    # Tạo cửa sổ mới
    new_window = tk.Toplevel(root)
    new_window.title("Linear Regression Plot")

    # Thêm biểu đồ vào cửa sổ mới
    canvas = FigureCanvasTkAgg(plt.gcf(), master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#
#
#LOGICTIS REGRESSION
def plot_logistic_regression():
    # Lấy tên biến phụ thuộc
    dependent_variable = get_target()

    # Lấy tên biến độc lập
    independent_variable = selected_input_variables_listbox.get(tk.ACTIVE)

    # Lấy dữ liệu
    X = df[[independent_variable]]
    y = df[dependent_variable]

    # Khởi tạo mô hình hồi quy logistic
    model = LogisticRegression()

    # Huấn luyện mô hình
    model.fit(X, y)

    # Dự đoán xác suất
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Vẽ biểu đồ scatter plot và đường hồi quy logistic
    plt.figure()
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred_proba, color='red')
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    plt.title(f"Logistic Regression: {dependent_variable} vs {independent_variable}")
    plt.grid(True)

    # Tạo cửa sổ mới
    new_window = tk.Toplevel(root)
    new_window.title("Logistic Regression Plot")

    # Thêm biểu đồ vào cửa sổ mới
    canvas = FigureCanvasTkAgg(plt.gcf(), master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


variables = []
selected_input_variables = []
root = tk.Tk()
root.title("CSV Variable Selector")
root.geometry("1300x700")

# Thay đổi màu sắc và giao diện cho các phần tử
root.config(bg="#f0f0f0")
left_Frame = tk.Frame(root, bg="#FDF5E6", width=300)
left_Frame.pack(side="left", fill="y")
load_button = ttk.Button(left_Frame, text="Tải Tệp CSV", command=load_csv_variables)
load_button.pack(pady=10)

info_label = tk.Label(left_Frame, text="Thông báo:", bg="#ffffff")
info_label.pack()
file_label = tk.Label(left_Frame, text="", fg="red", bg="#ffffff")
file_label.pack()
info_text = tk.Text(left_Frame, height=10, width=40)
info_text.pack(fill="both", padx=5, pady=5)

target_label = ttk.Label(left_Frame, text="Chọn Biến Mục Tiêu:", background="#ffffff")
target_label.pack(pady=5)
target_variable_combobox = ttk.Combobox(left_Frame, state="readonly")
target_variable_combobox.pack(pady=5)

input_label = ttk.Label(left_Frame, text="Chọn Biến Đầu Vào:", background="#ffffff")
input_label.pack(pady=5)

input_variable_listbox = tk.Listbox(left_Frame, selectmode=tk.MULTIPLE, height=3)
input_variable_listbox.pack(pady=5)

add_button = ttk.Button(left_Frame, text="Thêm", command=add_input_variable)
add_button.pack(pady=5)

selected_label = ttk.Label(left_Frame, text="Các Biến Đầu Vào Đã Chọn:", background="#ffffff")
selected_label.pack(pady=5)

selected_input_variables_listbox = tk.Listbox(left_Frame, selectmode=tk.MULTIPLE, height=3)
selected_input_variables_listbox.pack(pady=5)

remove_button = ttk.Button(left_Frame, text="Xóa", command=remove_input_variable)
remove_button.pack(pady=5)

status_label = ttk.Label(left_Frame, text="Chọn Mô Hình:", background="#ffffff")
status_label.pack(pady=10)
model_combobox = ttk.Combobox(left_Frame, values=["Hồi quy Tuyến tính", "Hồi quy Logistic", "KNN"], state="readonly")
model_combobox.pack(pady=5)

add_button = ttk.Button(left_Frame, text="Thực Thi", command=show_new_page)
add_button.pack(pady=5)

# Tạo frame chứa thanh chức năng và bảng dữ liệu
right_frame = tk.Frame(root, bg="#ffffff")
right_frame.pack(side="left", fill="both", expand=True)

# Tạo thanh chức năng
nav_frame = tk.Frame(right_frame, bg="#f0f0f0")
nav_frame.pack(fill="x")

info_button = tk.Button(nav_frame, text="Xem Thông Tin", command=show_info, relief=tk.FLAT, bg="#4CAF50", fg="white", padx=10)
info_button.pack(side="left", padx=5, pady=5)

view_data_button = tk.Button(nav_frame, text="Xem Dữ Liệu", command=show_data, relief=tk.FLAT, bg="#4CAF50", fg="white", padx=10)
view_data_button.pack(side="left", padx=5, pady=5)

process_button = tk.Button(nav_frame, text="Xử lý dữ liệu", command=process_missing_data, relief=tk.FLAT, bg="#4CAF50", fg="white", padx=10)
process_button.pack(side="left", padx=5, pady=5)

thongke_button = tk.Button(nav_frame, text="Biểu đồ thống kê", command=show_statistics, relief=tk.FLAT, bg="#4CAF50", fg="white", padx=10)
thongke_button.pack(side="left", padx=5, pady=5)

# Tạo bảng dữ liệu
data_table_frame = tk.Frame(right_frame)
data_table_frame.pack(fill="both", expand=True)

data_table = tk.Text(data_table_frame, wrap="none")
data_table.pack(side="top", fill="both", expand=True)

scrollbar_y = Scrollbar(data_table_frame, command=data_table.yview)
scrollbar_y.pack(side="right", fill="y")
data_table.config(yscrollcommand=scrollbar_y.set)

scrollbar_x = Scrollbar(data_table_frame, orient="horizontal", command=data_table.xview)
scrollbar_x.pack(side="bottom", fill="x")
data_table.config(xscrollcommand=scrollbar_x.set)

root.mainloop()
