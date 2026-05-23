import matplotlib.pyplot as plt
import numpy as np


def show_faces(images, titles=None, cols=6, figsize=(14, 8)):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        if titles:
            plt.title(titles[i], fontsize=8)
    plt.tight_layout()
    return plt.gcf()


def plot_ir_curves(results):
    plt.figure(figsize=(8, 5))
    for name, values in results.items():
        xs = sorted(values.keys(), reverse=True)
        ys = [values[x]["tpr"] for x in xs]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Identification Rate: TPR@FPR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()
