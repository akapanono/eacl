import os

import torch


def get_dataset_emotions(dataset_name):
    if dataset_name == "IEMOCAP":
        return ["neutral", "excited", "frustrated", "sad", "happy", "angry"]
    if dataset_name == "MELD":
        return ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    if dataset_name == "EmoryNLP":
        return ["joyful", "neutral", "powerful", "mad", "scared", "peaceful", "sad"]
    raise ValueError(f"Unsupported dataset for anchors: {dataset_name}")


def get_subanchor_templates(dataset_name):
    if dataset_name == "IEMOCAP":
        return {
            "neutral": [
                "The speaker feels neutral and steady.",
                "The speaker feels calm and emotionally balanced.",
                "The speaker feels matter-of-fact and composed.",
            ],
            "excited": [
                "The speaker feels excited and energized.",
                "The speaker feels thrilled and highly activated.",
                "The speaker feels enthusiastic and emotionally elevated.",
            ],
            "frustrated": [
                "The speaker feels frustrated and tense.",
                "The speaker feels blocked and irritated.",
                "The speaker feels upset because things are not working.",
            ],
            "sad": [
                "The speaker feels sad and down.",
                "The speaker feels disappointed and low in energy.",
                "The speaker feels hurt and emotionally withdrawn.",
            ],
            "happy": [
                "The speaker feels happy and warm.",
                "The speaker feels pleased and content.",
                "The speaker feels cheerful and positive.",
            ],
            "angry": [
                "The speaker feels angry and explosive.",
                "The speaker feels angry and tense.",
                "The speaker feels angry and irritated.",
            ],
        }
    if dataset_name == "MELD":
        return {
            "anger": [
                "The speaker feels angry and explosive.",
                "The speaker feels angry and tense.",
                "The speaker feels angry and irritated.",
            ],
            "disgust": [
                "The speaker feels disgusted and repelled.",
                "The speaker feels disgusted and uncomfortable.",
                "The speaker feels disgusted and rejecting.",
            ],
            "fear": [
                "The speaker feels fearful and tense.",
                "The speaker feels afraid and nervous.",
                "The speaker feels scared and alarmed.",
            ],
            "joy": [
                "The speaker feels joyful and excited.",
                "The speaker feels joyful and bright.",
                "The speaker feels joyful and socially warm.",
            ],
            "sadness": [
                "The speaker feels sad and withdrawn.",
                "The speaker feels sad and disappointed.",
                "The speaker feels sad and hurt.",
            ],
            "surprise": [
                "The speaker feels surprised and startled.",
                "The speaker feels surprised and curious.",
                "The speaker feels surprised and reactive.",
            ],
            "neutral": [
                "The speaker feels neutral and steady.",
                "The speaker feels calm and emotionally balanced.",
                "The speaker feels matter-of-fact and composed.",
            ],
        }
    if dataset_name == "EmoryNLP":
        return {
            "joyful": [
                "The speaker feels joyful and bright.",
                "The speaker feels joyful and excited.",
                "The speaker feels joyful and affectionate.",
            ],
            "neutral": [
                "The speaker feels neutral and steady.",
                "The speaker feels calm and emotionally balanced.",
                "The speaker feels matter-of-fact and composed.",
            ],
            "powerful": [
                "The speaker feels powerful and confident.",
                "The speaker feels powerful and assertive.",
                "The speaker feels powerful and in control.",
            ],
            "mad": [
                "The speaker feels mad and explosive.",
                "The speaker feels mad and irritated.",
                "The speaker feels mad and tense.",
            ],
            "scared": [
                "The speaker feels scared and tense.",
                "The speaker feels scared and nervous.",
                "The speaker feels scared and alarmed.",
            ],
            "peaceful": [
                "The speaker feels peaceful and relaxed.",
                "The speaker feels peaceful and settled.",
                "The speaker feels peaceful and emotionally safe.",
            ],
            "sad": [
                "The speaker feels sad and withdrawn.",
                "The speaker feels sad and disappointed.",
                "The speaker feels sad and hurt.",
            ],
        }
    raise ValueError(f"Unsupported dataset for templates: {dataset_name}")


def expand_templates(dataset_name, num_subanchors):
    templates = get_subanchor_templates(dataset_name)
    expanded = {}
    for emotion, variants in templates.items():
        if len(variants) >= num_subanchors:
            expanded[emotion] = variants[:num_subanchors]
            continue
        copied = list(variants)
        while len(copied) < num_subanchors:
            copied.append(f"{variants[len(copied) % len(variants)]} Variant {len(copied) + 1}.")
        expanded[emotion] = copied
    return expanded


def get_anchor_filename(dataset_name, num_subanchors):
    return f"{dataset_name.lower()}_emo_{num_subanchors}.pt"


def load_anchor_tensor(anchor_path, dataset_name, num_subanchors):
    preferred = os.path.join(anchor_path, get_anchor_filename(dataset_name, num_subanchors))
    if os.path.exists(preferred):
        anchors = torch.load(preferred, map_location="cpu")
    else:
        legacy = os.path.join(anchor_path, f"{dataset_name.lower()}_emo.pt")
        anchors = torch.load(legacy, map_location="cpu")
    if anchors.dim() == 2:
        anchors = anchors.unsqueeze(1)
    return anchors
