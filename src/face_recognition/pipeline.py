"""End-to-end inference helpers for the face recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from face_recognition.alignment import align_face, heatmaps_to_landmarks
from face_recognition.metrics import cosine_matrix
from face_recognition.models.embedding import EmbeddingNet, FaceClassifier
from face_recognition.models.hourglass import StackedHourglassNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PipelineResult:
    """Prediction result for one detected or supplied face."""

    source: str
    face_index: int
    landmarks: np.ndarray
    aligned_face: np.ndarray
    embedding: np.ndarray | None = None
    predictions: list[tuple[str, float]] | None = None
    nearest: list[tuple[str, float]] | None = None


def list_images(path: str | Path) -> list[Path]:
    """Return image files from a file or directory path."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_landmark_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    num_stacks: int = 2,
    num_blocks: int = 4,
    channels: int = 128,
    num_keypoints: int = 5,
) -> StackedHourglassNet:
    """Load a Stacked Hourglass landmark model from a checkpoint."""
    model = StackedHourglassNet(num_stacks=num_stacks, num_blocks=num_blocks, channels=channels, num_keypoints=num_keypoints)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_face_model(
    checkpoint: str | Path,
    model_type: str = "embedding",
    num_classes: int | None = None,
    embedding_dim: int = 512,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load an embedding or classifier model used by the recognition stage."""
    if model_type == "classifier":
        if num_classes is None:
            raise ValueError("num_classes is required for classifier checkpoints")
        model = FaceClassifier(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
    elif model_type == "embedding":
        model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
    else:
        raise ValueError("model_type must be either 'embedding' or 'classifier'")

    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload.get("encoder_state_dict", payload))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_gallery(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load gallery embeddings from an `.npz` file with `embeddings` and `labels`."""
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["embeddings"], dtype=np.float32), np.asarray(data["labels"])


def detect_faces(image: Image.Image, min_confidence: float = 0.96, margin: float = 0.3, device: str = "cpu") -> list[Image.Image]:
    """Detect face crops with MTCNN when available; otherwise return the full image."""
    try:
        from facenet_pytorch import MTCNN
    except ModuleNotFoundError:
        return [image.convert("RGB")]

    detector = MTCNN(keep_all=True, device=device)
    boxes, probs = detector.detect(image, landmarks=False)
    if boxes is None or probs is None:
        return [image.convert("RGB")]

    crops: list[Image.Image] = []
    width, height = image.size
    for box, prob in zip(boxes, probs):
        if prob is None or prob < min_confidence:
            continue
        x1, y1, x2, y2 = box.astype(float)
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        crop_box = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
        crops.append(image.crop(crop_box).convert("RGB"))
    return crops or [image.convert("RGB")]


def predict_landmarks(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray,
    device: str | torch.device = "cpu",
    image_size: int = 256,
) -> np.ndarray:
    """Predict five facial landmark coordinates for one face image."""
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        heatmaps = outputs[-1][0] if isinstance(outputs, list) else outputs[0]
    return heatmaps_to_landmarks(heatmaps, image_size=image_size)


def align_with_landmarks(image: Image.Image | np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Align one face image by landmark coordinates."""
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    return align_face(array, landmarks, output_size=output_size)


def extract_embedding(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray,
    device: str | torch.device = "cpu",
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return `(embedding, logits)` for one aligned face."""
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, "classifier"):
            logits, embedding = model(tensor, return_embedding=True)
            return embedding.squeeze(0).cpu().numpy(), logits.squeeze(0).cpu().numpy()
        embedding = model(tensor)
        return embedding.squeeze(0).cpu().numpy(), None


def topk_predictions(logits: np.ndarray, class_names: Iterable[str] | None = None, top_k: int = 5) -> list[tuple[str, float]]:
    """Convert classifier logits to top-k label probabilities."""
    logits = np.asarray(logits, dtype=np.float32)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    indices = np.argsort(probs)[::-1][:top_k]
    names = list(class_names) if class_names is not None else [str(i) for i in range(len(probs))]
    return [(names[i], float(probs[i])) for i in indices]


def nearest_identities(
    embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return nearest gallery identities by cosine similarity."""
    scores = cosine_matrix(np.asarray(embedding, dtype=np.float32)[None, :], gallery_embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]
    return [(str(gallery_labels[i]), float(scores[i])) for i in indices]


class FaceRecognitionPipeline:
    """Small end-to-end face recognition pipeline for inference and demos."""

    def __init__(
        self,
        landmark_model: torch.nn.Module,
        face_model: torch.nn.Module | None = None,
        device: str | torch.device = "cpu",
        landmark_image_size: int = 256,
        aligned_size: int = 112,
        embedding_image_size: int = 224,
        class_names: list[str] | None = None,
        gallery: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.landmark_model = landmark_model
        self.face_model = face_model
        self.device = device
        self.landmark_image_size = landmark_image_size
        self.aligned_size = aligned_size
        self.embedding_image_size = embedding_image_size
        self.class_names = class_names
        self.gallery = gallery

    def process_image(self, image_path: str | Path, detect: bool = True, top_k: int = 5) -> list[PipelineResult]:
        """Run detection, landmark prediction, alignment and optional recognition for one image."""
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        faces = detect_faces(image, device=str(self.device)) if detect else [image]
        results: list[PipelineResult] = []
        for index, face in enumerate(faces):
            face_for_landmarks = face.resize((self.landmark_image_size, self.landmark_image_size))
            landmarks = predict_landmarks(self.landmark_model, face_for_landmarks, self.device, self.landmark_image_size)
            aligned = align_with_landmarks(face_for_landmarks, landmarks, output_size=self.aligned_size)
            embedding = None
            predictions = None
            nearest = None
            if self.face_model is not None:
                embedding, logits = extract_embedding(self.face_model, aligned, self.device, self.embedding_image_size)
                if logits is not None:
                    predictions = topk_predictions(logits, self.class_names, top_k=top_k)
                if self.gallery is not None:
                    gallery_embeddings, gallery_labels = self.gallery
                    nearest = nearest_identities(embedding, gallery_embeddings, gallery_labels, top_k=top_k)
            results.append(
                PipelineResult(
                    source=str(image_path),
                    face_index=index,
                    landmarks=landmarks,
                    aligned_face=aligned,
                    embedding=embedding,
                    predictions=predictions,
                    nearest=nearest,
                )
            )
        return results

    def process_paths(self, paths: Iterable[str | Path], detect: bool = True, top_k: int = 5) -> list[PipelineResult]:
        """Run the pipeline for multiple image paths."""
        all_results: list[PipelineResult] = []
        for path in paths:
            all_results.extend(self.process_image(path, detect=detect, top_k=top_k))
        return all_results


def save_aligned_face(image: np.ndarray, output_path: str | Path) -> None:
    """Save an aligned RGB face image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
