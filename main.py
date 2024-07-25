import tkinter as tk
from gui import FacialLandmarkCollectorUI


if __name__ == "__main__":
    root = tk.Tk()
    app = FacialLandmarkCollectorUI(root)
    root.mainloop()