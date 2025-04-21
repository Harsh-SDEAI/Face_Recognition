import tkinter as tk
from tkinter import ttk

def update_label(value):
    """
    Update the label to display the mapped value corresponding to the slider position.
    """
    user_input = float(value)
    mapped_value = 0.5 + (user_input / 100) * (1.5 - 0.5)
    mapped_value_label.config(text=f"Mapped Value: {mapped_value:.2f}")

# Create the main application window
root = tk.Tk()
root.title("Threshold Mapping UI")
root.geometry("1920x1080")

# Add a label to display instructions
instruction_label = ttk.Label(root, text="Move the slider to set a value between 0 and 100:", font=("Arial", 12))
instruction_label.pack(pady=10)

# Add a frame to hold the slider and tick labels
slider_frame = ttk.Frame(root)
slider_frame.pack(pady=20)

# Create the slider widget
slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal", command=update_label, length=400)
slider.grid(row=1, column=0, padx=10)

# Add tick labels above the slider
tick_labels = ["0", "20", "40", "60", "80", "100"]
for i, tick in enumerate(tick_labels):
    tick_label = ttk.Label(slider_frame, text=tick, font=("Arial", 10))
    tick_label.grid(row=0, column=i, padx=2)

# Add a label to display the mapped value
mapped_value_label = ttk.Label(root, text="Mapped Value: 0.80", font=("Arial", 12))
mapped_value_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
