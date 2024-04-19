import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, font
import pandas as pd


class DataAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("app_Chí")

        self.df = None
        self.file_name = None

        # Tạo frame chứa nút load csv và cảnh báo
        self.left_frame = tk.Frame(self.root, width=300)
        self.left_frame.pack(side="left", fill="y")

        self.load_button = tk.Button(self.left_frame, text="Tải lên file CSV", command=self.load_csv)
        self.load_button.pack(pady=10)

        self.file_label = tk.Label(self.left_frame, text="", fg="red")
        self.file_label.pack()

        self.info_label = tk.Label(self.left_frame, text="Thông báo:")
        self.info_label.pack()

        self.info_text = tk.Text(self.left_frame, height=10, width=40)
        self.info_text.pack(fill="both", padx=5, pady=5)

        # Tạo frame chứa thanh chức năng và bảng dữ liệu
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side="left", fill="both", expand=True)

        # Tạo thanh chức năng
        self.nav_frame = tk.Frame(self.right_frame)
        self.nav_frame.pack(fill="x")

        self.info_button = tk.Button(self.nav_frame, text="Xem Thông Tin", command=self.display_info)
        self.info_button.pack(side="left", padx=5, pady=5)

        self.num_vars_button = tk.Button(self.nav_frame, text="Xem Biến Số", command=self.display_numerical_variables)
        self.num_vars_button.pack(side="left", padx=5, pady=5)

        self.cat_vars_button = tk.Button(self.nav_frame, text="Xem Biến Phân Loại",
                                         command=self.display_categorical_variables)
        self.cat_vars_button.pack(side="left", padx=5, pady=5)

        self.view_data_button = tk.Button(self.nav_frame, text="Xem Dữ Liệu", command=self.display_data)
        self.view_data_button.pack(side="left", padx=5, pady=5)

        # Tạo bảng dữ liệu
        self.data_table_frame = tk.Frame(self.right_frame)
        self.data_table_frame.pack(fill="both", expand=True)

        self.data_table = tk.Text(self.data_table_frame, wrap="none")
        self.data_table.pack(side="top", fill="both", expand=True)

        self.scrollbar_y = Scrollbar(self.data_table_frame, command=self.data_table.yview)
        self.scrollbar_y.pack(side="right", fill="y")
        self.data_table.config(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_x = Scrollbar(self.data_table_frame, orient="horizontal", command=self.data_table.xview)
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.data_table.config(xscrollcommand=self.scrollbar_x.set)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path, encoding='ISO-8859-1')
                self.file_name = file_path.split('/')[-1]  # Lấy tên file từ đường dẫn
                self.file_label.config(text=self.file_name)  # Hiển thị tên file
                messagebox.showinfo("Thành công", "File CSV đã được tải thành công.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải file CSV: {e}")

    def display_info(self):
        if self.df is not None:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, str(self.df.info()))
        else:
            messagebox.showerror("Lỗi", "Không có file CSV nào được tải.")

    def display_numerical_variables(self):
        if self.df is not None:
            numerical_vars = self.df.select_dtypes(include='number').columns
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Biến Số:\n")
            self.info_text.insert(tk.END, "\n".join(numerical_vars))
        else:
            messagebox.showerror("Lỗi", "Không có file CSV nào được tải.")

    def display_categorical_variables(self):
        if self.df is not None:
            categorical_vars = self.df.select_dtypes(include='object').columns
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Biến Phân Loại:\n")
            self.info_text.insert(tk.END, "\n".join(categorical_vars))
        else:
            messagebox.showerror("Lỗi", "Không có file CSV nào được tải.")

    def display_data(self):
        if self.df is not None:
            if 'Name' in self.df.columns:
                # Tạo một bản sao của DataFrame để giảm kích thước cột "Name"
                df_copy = self.df.copy()
                df_copy["Name"] = df_copy["Name"].str[:20]  # Giới hạn độ dài của cột "Name" thành 20 ký tự
            else:
                df_copy = self.df  # Sử dụng DataFrame gốc nếu không có cột "Name"
            self.data_table.delete(1.0, tk.END)
            self.data_table.insert(tk.END, df_copy.to_string(index=False))
        else:
            messagebox.showerror("Lỗi", "Không có file CSV nào được tải.")


def main():
    root = tk.Tk()
    app = DataAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
