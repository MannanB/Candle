"""Draw a digit and try the saved MNIST binary classifier."""

import sys
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candle

TARGET_NUM = 2
THRESHOLD = 0.3
MODEL_PATH = Path(__file__).resolve().parent / f"mnist_classifier_{TARGET_NUM}"
CANVAS_SIZE = 280
BRUSH_SIZE = 20


class BinaryClassifier:
    def __init__(self, path):
        self.l1 = candle.Linear.load(str(path / "l1.bin"))
        self.l2 = candle.Linear.load(str(path / "l2.bin"))
        for layer in (self.l1, self.l2):
            layer.weights.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False

    def forward(self, x):
        return self.l2.forward(candle.activations.relu(self.l1.forward(x)))


def prepare_input(image):
    """Fit the digit into a centered 20x20 box on MNIST's 28x28 canvas."""
    digit = image.crop(image.getbbox())
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    pixels = list(centered.getdata())
    rows = [pixels[i:i + 28] for i in range(0, 784, 28)]
    return candle.Tensor([rows]).flatten(1, 2).unsqueeze() * (1 / 255.0)


class DrawingWindow:
    def __init__(self, root, classifier):
        self.classifier = classifier
        self.root = root
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.previous = None
        root.title(f"MNIST: is it a {TARGET_NUM}?")
        tk.Label(root, text="Draw a digit. Release the mouse to predict.").pack(pady=8)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=0)
        self.canvas.pack(padx=12)
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.finish_stroke)
        self.result = tk.StringVar(value=f"Draw a {TARGET_NUM} or another digit")
        tk.Label(root, textvariable=self.result).pack(pady=8)
        buttons = tk.Frame(root)
        buttons.pack(pady=(0, 12))
        tk.Button(buttons, text="Predict", command=self.predict).pack(side="left", padx=4)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=4)
        root.bind("<Return>", lambda event: self.predict())
        root.bind("<Escape>", lambda event: self.clear())

    def start_stroke(self, event):
        self.previous = None
        self.paint(event)

    def paint(self, event):
        x = max(0, min(CANVAS_SIZE - 1, event.x))
        y = max(0, min(CANVAS_SIZE - 1, event.y))
        radius = BRUSH_SIZE / 2
        if self.previous is not None:
            x0, y0 = self.previous
            self.canvas.create_line(x0, y0, x, y, fill="white", width=BRUSH_SIZE,
                                    capstyle=tk.ROUND)
            self.draw.line((x0, y0, x, y), fill=255, width=BRUSH_SIZE)
        box = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*box, fill="white", outline="white")
        self.draw.ellipse(box, fill=255)
        self.previous = (x, y)

    def finish_stroke(self, event):
        self.previous = None
        self.predict()

    def predict(self):
        if self.image.getbbox() is None:
            self.result.set("Draw a digit first")
            return
        score = self.classifier.forward(prepare_input(self.image)).tolist()[0][0][0]
        verdict = f"IS {TARGET_NUM}" if score > THRESHOLD else f"NOT {TARGET_NUM}"
        message = f"{verdict} (score: {score:.3f})"
        self.result.set(message)
        print(message, flush=True)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.previous = None
        self.result.set(f"Draw a {TARGET_NUM} or another digit")


if __name__ == "__main__":
    classifier = BinaryClassifier(MODEL_PATH)
    root = tk.Tk()
    DrawingWindow(root, classifier)
    root.mainloop()
