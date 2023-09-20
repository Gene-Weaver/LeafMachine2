import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
from pandastable import Table, TableModel
import numpy as np

class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid(sticky='nsew')
        self.current_row = None

        # Set grid expansion properties
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure([0,1,2], weight=1)

        # Set dark theme
        self.master.tk_setPalette(background='#2B2B2B', foreground='white')

        # File name and length of rows
        self.file_info = tk.Label(self, text="File name: unknown.xlsx, Rows: 0")
        self.file_info.grid(row=0, column=0, sticky='nsew')

        # Buttons for moving to the next or previous rows
        self.prev_button = tk.Button(self, text="Previous", command=self.prev_row)
        self.next_button = tk.Button(self, text="Next", command=self.next_row)
        self.load_button = tk.Button(self, text="Load XLSX", command=self.load_xlsx)

        self.prev_button.grid(row=0, column=0, sticky='w')
        self.next_button.grid(row=0, column=0)
        self.load_button.grid(row=0, column=0, sticky='e')

        # Outer frame
        self.outer_frame = tk.Frame(self)
        self.outer_frame.grid(row=1, column=0, sticky='nsew')
        # Set expansion properties for the outer frame
        self.outer_frame.columnconfigure(0, weight=1)
        self.outer_frame.rowconfigure(0, weight=1)

        # Frame for XLSX content
        self.xlsx_frame = tk.Frame(self.outer_frame)
        self.xlsx_frame.pack(fill='both', expand=True)

        # Frame for row tracker with a fixed width
        self.tracker_frame = tk.Frame(self.xlsx_frame, width=40, bg='red')  # added a red background for clarity
        self.tracker_frame.pack(side='left', fill='y')
        self.tracker_frame.pack_propagate(False)  # prevents the frame from resizing to fit its content

        # Text area for the current xlsx row inside the tracker frame
        self.xlsx_text = tk.Text(self.tracker_frame)
        self.xlsx_text.pack(fill='both', expand=True)  # makes the text widget fill the frame

        # Frame for the table
        self.table_frame = tk.Frame(self.xlsx_frame)
        self.table_frame.pack(side='left', fill='both', expand=True)

        # Set grid expansion properties
        self.xlsx_frame.columnconfigure(0, weight=1)
        self.xlsx_frame.rowconfigure(0, weight=1)

        # Outer frame for the JSON components
        self.outer_json_frame = tk.Frame(self)
        self.outer_json_frame.grid(row=2, column=0, sticky='nsew')
        # Set expansion properties for the JSON outer frame
        self.outer_json_frame.columnconfigure(0, weight=1)
        self.outer_json_frame.rowconfigure(0, weight=1)

        # Frame for JSON text
        self.json_frame1 = tk.Frame(self.outer_json_frame)
        self.json_frame2 = tk.Frame(self.outer_json_frame)
        self.json_frame1.pack(side='left', fill='both', expand=True)
        self.json_frame2.pack(side='left', fill='both', expand=True)

        # Text area for JSON packet
        self.json_text1 = tk.Text(self.json_frame1)
        self.json_text2 = tk.Text(self.json_frame2)
        self.json_text1.pack(fill='both', expand=True)
        self.json_text2.pack(fill='both', expand=True)

        # Create the image window and canvas
        self.image_window = tk.Toplevel(self)
        self.image_window.title("Image Window")
        self.canvas = tk.Canvas(self.image_window, width=self.master.winfo_screenwidth(), height=1080)

        # Create a black image by default
        self.photo = ImageTk.PhotoImage(image=Image.new("RGB", (self.master.winfo_screenwidth(), 1080), "black"))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Configure the scrollregion to the size of the image
        self.canvas.config(scrollregion=(0, 0, self.master.winfo_screenwidth(), 1080))

        # Add vertical scrollbar
        vscrollbar = tk.Scrollbar(self.image_window, command=self.canvas.yview)
        vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar
        hscrollbar = tk.Scrollbar(self.image_window, command=self.canvas.xview, orient='horizontal')
        hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.config(yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


    def prev_row(self):
        # Go to the previous row
        pass

    def next_row(self):
        # Go to the next row
        pass

    def load_xlsx(self):
        # Open file dialog to select xlsx file
        file_path = filedialog.askopenfilename(filetypes=[("XLSX files", "*.xlsx")])
        if file_path:
            self.file_info.config(text=f"File name: {file_path}, Rows: 0")
            # Load xlsx file
            df = pd.read_excel(file_path)
            self.show_dataframe(df)

    def show_dataframe(self, df):
        # Create table canvas
        self.pt = Table(self.table_frame, dataframe=df, showtoolbar=True, showstatusbar=True)

        # Bind <ButtonRelease-1> event to self.on_table_click
        self.pt.bind('<ButtonRelease-1>', self.on_table_click)

        # Bind "Up" and "Down" keys
        self.pt.bind('<Up>', self.on_up_key)
        self.pt.bind('<Down>', self.on_down_key)

        self.pt.show()

        # Select the first row
        self.pt.setSelectedRow(0)
        self.current_row = 0
        self.load_image_from_table()

    def on_up_key(self, event):
        # Decrement current_row, ensuring it does not go below 0
        self.current_row = max(0, self.current_row - 1)
        self.pt.setSelectedRow(self.current_row)
        self.pt.redraw()  # replace with the correct method name
        self.load_image_from_table()

    def on_down_key(self, event):
        # Increment current_row, ensuring it does not exceed the last row
        self.current_row = min(self.pt.model.getRowCount() - 1, self.current_row + 1)
        self.pt.setSelectedRow(self.current_row)
        self.pt.redraw()  # replace with the correct method name
        self.load_image_from_table()


    def on_table_click(self, event=None):
        # If the selection has changed, update the active row and load the new image
        current_selection = self.pt.getSelectedRow()
        if current_selection != self.current_row:
            self.current_row = current_selection
            self.load_image_from_table()

    def load_image_from_table(self):
        # Load image from 'path_to_crop' column in the active row
        if self.current_row is not None and 'path_to_crop' in self.pt.model.df.columns:
            path_to_crop = self.pt.model.df.iloc[self.current_row]['path_to_crop']
            print(path_to_crop)
            self.display_image(path_to_crop)

    def display_image(self, image_path):
        # Open the image and convert it for tkinter
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)

        # Update the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Configure the scrollregion to the size of the image
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


root = tk.Tk()
app = Application(master=root)
app.mainloop()