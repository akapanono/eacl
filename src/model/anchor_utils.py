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
                "The speaker feels neutral and controlled.",
                "The speaker feels emotionally even and unreactive.",
                "The speaker feels neutral and observational.",
            ],
            "excited": [
                "The speaker feels excited and energized.",
                "The speaker feels thrilled and highly activated.",
                "The speaker feels enthusiastic and emotionally elevated.",
                "The speaker feels excited and eager.",
                "The speaker feels animated and intensely positive.",
                "The speaker feels excited and expressive.",
            ],
            "frustrated": [
                "The speaker feels frustrated and tense.",
                "The speaker feels blocked and irritated.",
                "The speaker feels upset because things are not working.",
                "The speaker feels frustrated and impatient.",
                "The speaker feels annoyed and constrained.",
                "The speaker feels frustrated and discouraged.",
            ],
            "sad": [
                "The speaker feels sad and down.",
                "The speaker feels disappointed and low in energy.",
                "The speaker feels hurt and emotionally withdrawn.",
                "The speaker feels sad and lonely.",
                "The speaker feels sorrowful and discouraged.",
                "The speaker feels sad and helpless.",
            ],
            "happy": [
                "The speaker feels happy and warm.",
                "The speaker feels pleased and content.",
                "The speaker feels cheerful and positive.",
                "The speaker feels happy and relaxed.",
                "The speaker feels delighted and lighthearted.",
                "The speaker feels happy and socially open.",
            ],
            "angry": [
                "The speaker feels angry and explosive.",
                "The speaker feels angry and tense.",
                "The speaker feels angry and irritated.",
                "The speaker feels angry and confrontational.",
                "The speaker feels angry and offended.",
                "The speaker feels angry and agitated.",
            ],
        }
    if dataset_name == "MELD":
        return {
            "anger": [
                "The speaker feels angry and explosive.",
                "The speaker feels angry and tense.",
                "The speaker feels angry and irritated.",
                "The speaker feels angry and confrontational.",
                "The speaker feels angry and offended.",
                "The speaker feels angry and agitated.",
            ],
            "disgust": [
                "The speaker feels disgusted and repelled.",
                "The speaker feels disgusted and uncomfortable.",
                "The speaker feels disgusted and rejecting.",
                "The speaker feels disgusted and revolted.",
                "The speaker feels disgusted and dismissive.",
                "The speaker feels disgusted and averse.",
            ],
            "fear": [
                "The speaker feels fearful and tense.",
                "The speaker feels afraid and nervous.",
                "The speaker feels scared and alarmed.",
                "The speaker feels fearful and uncertain.",
                "The speaker feels frightened and defensive.",
                "The speaker feels scared and overwhelmed.",
            ],
            "joy": [
                "The speaker feels joyful and excited.",
                "The speaker feels joyful and bright.",
                "The speaker feels joyful and socially warm.",
                "The speaker feels joyful and playful.",
                "The speaker feels joyful and relaxed.",
                "The speaker feels joyful and delighted.",
            ],
            "sadness": [
                "The speaker feels sad and withdrawn.",
                "The speaker feels sad and disappointed.",
                "The speaker feels sad and hurt.",
                "The speaker feels sad and lonely.",
                "The speaker feels sad and helpless.",
                "The speaker feels sad and discouraged.",
            ],
            "surprise": [
                "The speaker feels surprised and startled.",
                "The speaker feels surprised and curious.",
                "The speaker feels surprised and reactive.",
                "The speaker feels surprised and amazed.",
                "The speaker feels surprised and confused.",
                "The speaker feels surprised and caught off guard.",
            ],
            "neutral": [
                "The speaker feels neutral and steady.",
                "The speaker feels calm and emotionally balanced.",
                "The speaker feels matter-of-fact and composed.",
                "The speaker feels neutral and controlled.",
                "The speaker feels emotionally even and unreactive.",
                "The speaker feels neutral and observational.",
            ],
        }
    if dataset_name == "EmoryNLP":
        return {
            "joyful": [
                "The speaker feels joyful and bright.",
                "The speaker feels joyful and excited.",
                "The speaker feels joyful and affectionate.",
                "The speaker feels joyful and playful.",
                "The speaker feels joyful and relaxed.",
                "The speaker feels joyful and delighted.",
            ],
            "neutral": [
                "The speaker feels neutral and steady.",
                "The speaker feels calm and emotionally balanced.",
                "The speaker feels matter-of-fact and composed.",
                "The speaker feels neutral and controlled.",
                "The speaker feels emotionally even and unreactive.",
                "The speaker feels neutral and observational.",
            ],
            "powerful": [
                "The speaker feels powerful and confident.",
                "The speaker feels powerful and assertive.",
                "The speaker feels powerful and in control.",
                "The speaker feels powerful and self-assured.",
                "The speaker feels powerful and commanding.",
                "The speaker feels powerful and dominant.",
            ],
            "mad": [
                "The speaker feels mad and explosive.",
                "The speaker feels mad and irritated.",
                "The speaker feels mad and tense.",
                "The speaker feels mad and confrontational.",
                "The speaker feels mad and offended.",
                "The speaker feels mad and agitated.",
            ],
            "scared": [
                "The speaker feels scared and tense.",
                "The speaker feels scared and nervous.",
                "The speaker feels scared and alarmed.",
                "The speaker feels scared and uncertain.",
                "The speaker feels scared and defensive.",
                "The speaker feels scared and overwhelmed.",
            ],
            "peaceful": [
                "The speaker feels peaceful and relaxed.",
                "The speaker feels peaceful and settled.",
                "The speaker feels peaceful and emotionally safe.",
                "The speaker feels peaceful and composed.",
                "The speaker feels peaceful and quietly content.",
                "The speaker feels peaceful and untroubled.",
            ],
            "sad": [
                "The speaker feels sad and withdrawn.",
                "The speaker feels sad and disappointed.",
                "The speaker feels sad and hurt.",
                "The speaker feels sad and lonely.",
                "The speaker feels sad and helpless.",
                "The speaker feels sad and discouraged.",
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
