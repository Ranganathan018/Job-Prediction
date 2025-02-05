import requests
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a Resume PDF", filetypes=[("PDF Files", "*.pdf")])
    return file_path

resume_path = select_file()
if not resume_path:
    print("No file selected. Exiting.")
    exit()

url = "http://127.0.0.1:8000/match-jobs/"
with open(resume_path, "rb") as file:
    files = {"resume": file}
    response = requests.post(url, files=files)

print(response.json())
