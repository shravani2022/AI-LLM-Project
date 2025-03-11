# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns

# class ImageToMatrixConverter:
#     def _init_(self):
#         self.root = tk.Tk()
#         self.root.title("Image to Matrix Converter")
#         self.root.geometry("400x200")
        
#         # Create and pack UI elements
#         self.create_ui()
        
#         self.image_data = None
#         self.filename = None
    
#     def create_ui(self):
#         # Create main frame
#         main_frame = tk.Frame(self.root, padx=20, pady=20)
#         main_frame.pack(expand=True, fill='both')
        
#         # Add title label
#         title_label = tk.Label(
#             main_frame,
#             text="Select a PNG image to convert to matrix",
#             font=("Arial", 12)
#         )
#         title_label.pack(pady=10)
        
#         # Add select file button
#         select_button = tk.Button(
#             main_frame,
#             text="Select Image",
#             command=self.select_file,
#             width=20
#         )
#         select_button.pack(pady=10)
        
#         # Add convert button
#         convert_button = tk.Button(
#             main_frame,
#             text="Convert and Visualize",
#             command=self.process_image,
#             width=20
#         )
#         convert_button.pack(pady=10)
        
#         # Add status label
#         self.status_label = tk.Label(
#             main_frame,
#             text="No file selected",
#             font=("Arial", 10)
#         )
#         self.status_label.pack(pady=10)
    
#     def select_file(self):
#         """Open file dialog and store selected filename"""
#         self.filename = filedialog.askopenfilename(
#             filetypes=[("PNG files", "*.png")]
#         )
#         if self.filename:
#             self.status_label.config(
#                 text=f"Selected: {self.filename.split('/')[-1]}"
#             )
    
#     def print_matrix_grid(self, matrix, chunk_size=30):
#         """Print matrix with row and column indices in a grid format with chunking"""
#         rows, cols = matrix.shape
        
#         # Calculate column width based on maximum number width
#         col_width = max(
#             len(str(matrix.max())),
#             len(str(matrix.min())),
#             len(str(cols))
#         ) + 1
        
#         # Print matrix in chunks
#         for start_row in range(0, rows, chunk_size):
#             for start_col in range(0, cols, chunk_size):
#                 end_row = min(start_row + chunk_size, rows)
#                 end_col = min(start_col + chunk_size, cols)
                
#                 # Print chunk header
#                 print(f"\nMatrix Chunk [{start_row}:{end_row}, {start_col}:{end_col}]:")
                
#                 # Print column headers for this chunk
#                 print(" " * (len(str(rows)) + 2), end="")
#                 for j in range(start_col, end_col):
#                     print(f"{j:>{col_width}}", end="")
#                 print("\n" + "-" * (col_width * (end_col - start_col) + len(str(rows)) + 2))
                
#                 # Print each row in the chunk
#                 for i in range(start_row, end_row):
#                     print(f"{i:>{len(str(rows))}} |", end="")
#                     for j in range(start_col, end_col):
#                         print(f"{matrix[i,j]:>{col_width}}", end="")
#                     print()
#                 print()  # Add space between chunks
    
#     def process_image(self):
#         """Convert image to matrix and visualize"""
#         if not self.filename:
#             self.status_label.config(text="Please select an image first!")
#             return
        
#         # Load and convert image to grayscale
#         img = Image.open(self.filename).convert('L')
        
#         # Calculate new dimensions maintaining aspect ratio
#         max_size = 100
#         aspect_ratio = img.size[0] / img.size[1]
#         if img.size[0] > max_size or img.size[1] > max_size:
#             if aspect_ratio > 1:
#                 new_size = (max_size, int(max_size / aspect_ratio))
#             else:
#                 new_size = (int(max_size * aspect_ratio), max_size)
#             img = img.resize(new_size, Image.LANCZOS)
        
#         self.image_data = np.array(img)
        
#         # Print matrix information
#         print(f"\nImage Matrix Information:")
#         print(f"Shape: {self.image_data.shape}")
#         print(f"Min value: {self.image_data.min()}")
#         print(f"Max value: {self.image_data.max()}")
#         print(f"Mean value: {self.image_data.mean():.2f}")
#         print(f"Standard deviation: {self.image_data.std():.2f}")
        
#         # Print the matrix grid
#         self.print_matrix_grid(self.image_data)
        
#         # Create visualizations
#         self.create_visualizations()
    
#     def create_visualizations(self):
#         """Create multiple visualizations of the matrix data"""
#         if self.image_data is None:
#             return
        
#         # Create a figure with multiple subplots
#         plt.style.use('dark_background')
#         fig = plt.figure(figsize=(18, 6))
        
#         # Original grayscale image with grid
#         ax1 = fig.add_subplot(131)
#         im1 = ax1.imshow(self.image_data, cmap='gray')
#         ax1.set_title('Original Grayscale Image')
#         # Add grid lines
#         ax1.grid(True, which='both', color='red', linestyle='-', linewidth=0.5, alpha=0.3)
#         ax1.set_xticks(np.arange(-.5, self.image_data.shape[1], 1))
#         ax1.set_yticks(np.arange(-.5, self.image_data.shape[0], 1))
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
        
#         # Heatmap with values and grid
#         ax2 = fig.add_subplot(132)
#         sns.heatmap(
#             self.image_data,
#             ax=ax2,
#             cmap='viridis',
#             xticklabels=True,
#             yticklabels=True,
#             cbar_kws={'label': 'Pixel Value'},
#             annot=True,
#             fmt='d',
#             annot_kws={'size': 6}
#         )
#         ax2.set_title('Pixel Values Heatmap')
        
#         # 3D surface plot with grid
#         ax3 = fig.add_subplot(133, projection='3d')
#         x, y = np.meshgrid(
#             np.arange(self.image_data.shape[1]),
#             np.arange(self.image_data.shape[0])
#         )
#         surf = ax3.plot_surface(
#             x, y,
#             self.image_data,
#             cmap='viridis',
#             linewidth=0.5,
#             alpha=0.8,
#             edgecolor='white'
#         )
#         ax3.set_title('3D Surface Plot of Pixel Values')
#         ax3.view_init(elev=30, azim=45)  # Adjust viewing angle
#         fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='Pixel Value')
        
#         plt.tight_layout(pad=3.0)
#         plt.show()

#     def run(self):
#         """Start the application"""
#         self.root.mainloop()

# if __name__ == "_main_":
#     app = ImageToMatrixConverter()
#     app.run()

# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk

# class ImageToMatrixConverter:
#     def __init__(self):
#         self.root = tk.Tk()
#         self.root.title("Image to Matrix Converter")
#         self.setup_ui()

#     def setup_ui(self):
#         main_frame = tk.Frame(self.root)
#         main_frame.pack(pady=20, padx=20)

#         title_label = tk.Label(
#             main_frame,
#             text="Select a PNG image to convert to matrix",
#             font=("Arial", 12)
#         )
#         title_label.pack(pady=10)
        
#         # Add select file button
#         select_button = tk.Button(
#             main_frame,
#             text="Select Image",
#             command=self.select_file
#         )
#         select_button.pack(pady=10)

#         self.image_label = tk.Label(main_frame)
#         self.image_label.pack(pady=10)

#     def select_file(self):
#         file_path = filedialog.askopenfilename(
#             filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
#         )
#         if file_path:
#             self.load_image(file_path)

#     def load_image(self, file_path):
#         image = Image.open(file_path)
#         image = image.resize((250, 250), Image.ANTIALIAS)
#         photo = ImageTk.PhotoImage(image)
#         self.image_label.config(image=photo)
#         self.image_label.image = photo

#     def run(self):
#         self.root.mainloop()

# if __name__ == "__main__":
#     app = ImageToMatrixConverter()
#     app.run()


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

class ImageToMatrixConverter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image to Matrix Converter")
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20, padx=20)

        title_label = tk.Label(
            main_frame,
            text="Select a PNG image to convert to matrix",
            font=("Arial", 12)
        )
        title_label.pack(pady=10)
        
        # Add select file button
        select_button = tk.Button(
            main_frame,
            text="Select Image",
            command=self.select_file
        )
        select_button.pack(pady=10)

        self.image_label = tk.Label(main_frame)
        self.image_label.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((250, 250), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageToMatrixConverter()
    app.run()