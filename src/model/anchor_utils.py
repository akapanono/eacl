import os

import torch

DOMAIN_NAMES = ["activation", "interaction", "expression", "context_shift"]


def get_dataset_emotions(dataset_name):
    if dataset_name == "IEMOCAP":
        return ["neutral", "excited", "frustrated", "sad", "happy", "angry"]
    if dataset_name == "MELD":
        return ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    if dataset_name == "EmoryNLP":
        return ["joyful", "neutral", "powerful", "mad", "scared", "peaceful", "sad"]
    raise ValueError(f"Unsupported dataset for anchors: {dataset_name}")


def get_domain_subanchor_templates(dataset_name):
    if dataset_name == "IEMOCAP":
        return {
            "neutral": [
                "The speaker feels neutral with low activation and steady energy.",
                "The speaker feels neutral in a socially balanced interaction.",
                "The speaker expresses neutrality in a controlled and even manner.",
                "The speaker remains neutral without a strong emotional shift from context.",
            ],
            "excited": [
                "The speaker feels excited with high activation and strong energy.",
                "The speaker feels excited while being socially engaged with others.",
                "The speaker expresses excitement openly and energetically.",
                "The speaker becomes excited as the dialogue context turns positive or stimulating.",
            ],
            "frustrated": [
                "The speaker feels frustrated with tension and blocked energy.",
                "The speaker feels frustrated in a resistant or dissatisfied interaction.",
                "The speaker expresses frustration impatiently but with some constraint.",
                "The speaker becomes frustrated because obstacles build up in the context.",
            ],
            "sad": [
                "The speaker feels sad with low energy and emotional heaviness.",
                "The speaker feels sad while withdrawing from the interaction or seeking support.",
                "The speaker expresses sadness in a subdued and hurt manner.",
                "The speaker becomes sad because the previous context creates loss or disappointment.",
            ],
            "happy": [
                "The speaker feels happy with positive and relaxed activation.",
                "The speaker feels happy in a warm and socially open interaction.",
                "The speaker expresses happiness cheerfully and openly.",
                "The speaker stays happy as the dialogue context remains positive.",
            ],
            "angry": [
                "The speaker feels angry with high tension and strong activation.",
                "The speaker feels angry in a confrontational or defensive interaction.",
                "The speaker expresses anger directly and with agitation.",
                "The speaker becomes angry because the dialogue context triggers conflict.",
            ],
        }
    if dataset_name == "MELD":
        return {
            "anger": [
                "The speaker feels anger with high tension and strong activation.",
                "The speaker feels anger in a confrontational or defensive interaction.",
                "The speaker expresses anger directly and with agitation.",
                "The speaker becomes angry because the dialogue context triggers conflict.",
            ],
            "disgust": [
                "The speaker feels disgust with aversive activation and rejection.",
                "The speaker feels disgust while distancing from another person or situation.",
                "The speaker expresses disgust through rejection or dismissiveness.",
                "The speaker becomes disgusted because the context presents something offensive.",
            ],
            "fear": [
                "The speaker feels fear with tense and alert activation.",
                "The speaker feels fear in a defensive or uncertain interaction.",
                "The speaker expresses fear nervously or cautiously.",
                "The speaker becomes fearful because the context suggests threat or uncertainty.",
            ],
            "joy": [
                "The speaker feels joy with positive and lively activation.",
                "The speaker feels joy in a warm and socially connected interaction.",
                "The speaker expresses joy brightly and playfully.",
                "The speaker becomes joyful as the dialogue context turns positive.",
            ],
            "sadness": [
                "The speaker feels sadness with low energy and emotional heaviness.",
                "The speaker feels sadness while withdrawing from the interaction or seeking support.",
                "The speaker expresses sadness in a subdued and hurt manner.",
                "The speaker becomes sad because the context creates loss or disappointment.",
            ],
            "surprise": [
                "The speaker feels surprise with sudden and reactive activation.",
                "The speaker feels surprise while reacting to another person or event.",
                "The speaker expresses surprise in a startled or curious manner.",
                "The speaker becomes surprised because the dialogue context changes unexpectedly.",
            ],
            "neutral": [
                "The speaker feels neutral with low activation and steady energy.",
                "The speaker feels neutral in a socially balanced interaction.",
                "The speaker expresses neutrality in a controlled and even manner.",
                "The speaker remains neutral without a strong emotional shift from context.",
            ],
        }
    if dataset_name == "EmoryNLP":
        return {
            "joyful": [
                "The speaker feels joyful with positive and lively activation.",
                "The speaker feels joyful in a warm and socially connected interaction.",
                "The speaker expresses joy brightly and playfully.",
                "The speaker becomes joyful as the dialogue context turns positive.",
            ],
            "neutral": [
                "The speaker feels neutral with low activation and steady energy.",
                "The speaker feels neutral in a socially balanced interaction.",
                "The speaker expresses neutrality in a controlled and even manner.",
                "The speaker remains neutral without a strong emotional shift from context.",
            ],
            "powerful": [
                "The speaker feels powerful with confident and controlled activation.",
                "The speaker feels powerful in an assertive or dominant interaction.",
                "The speaker expresses power in a confident and commanding manner.",
                "The speaker becomes powerful because the context gives control or advantage.",
            ],
            "mad": [
                "The speaker feels mad with high tension and strong activation.",
                "The speaker feels mad in a confrontational or defensive interaction.",
                "The speaker expresses madness directly and with agitation.",
                "The speaker becomes mad because the dialogue context triggers conflict.",
            ],
            "scared": [
                "The speaker feels scared with tense and alert activation.",
                "The speaker feels scared in a defensive or uncertain interaction.",
                "The speaker expresses fear nervously or cautiously.",
                "The speaker becomes scared because the context suggests threat or uncertainty.",
            ],
            "peaceful": [
                "The speaker feels peaceful with low tension and relaxed activation.",
                "The speaker feels peaceful in a safe and cooperative interaction.",
                "The speaker expresses peacefulness calmly and gently.",
                "The speaker remains peaceful because the context stays stable and safe.",
            ],
            "sad": [
                "The speaker feels sad with low energy and emotional heaviness.",
                "The speaker feels sad while withdrawing from the interaction or seeking support.",
                "The speaker expresses sadness in a subdued and hurt manner.",
                "The speaker becomes sad because the context creates loss or disappointment.",
            ],
        }
    raise ValueError(f"Unsupported dataset for domain templates: {dataset_name}")


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
    templates = get_domain_subanchor_templates(dataset_name)
    fallback_templates = get_subanchor_templates(dataset_name)
    expanded = {}
    for emotion, variants in templates.items():
        if len(variants) >= num_subanchors:
            expanded[emotion] = variants[:num_subanchors]
            continue
        copied = list(variants)
        while len(copied) < num_subanchors:
            fallback = fallback_templates[emotion][len(copied) % len(fallback_templates[emotion])]
            copied.append(fallback)
        expanded[emotion] = copied
    return expanded


def get_anchor_filename(dataset_name, num_subanchors):
    return f"{dataset_name.lower()}_emo_{num_subanchors}.pt"


def load_anchor_tensor(anchor_path, dataset_name, num_subanchors):
    preferred = os.path.join(anchor_path, get_anchor_filename(dataset_name, num_subanchors))
    if os.path.exists(preferred):
        anchors = torch.load(preferred, map_location="cpu")
    else:
        if num_subanchors > 1:
            raise FileNotFoundError(
                f"Missing anchor file: {preferred}. "
                f"Please run `python src/generate_anchors.py --bert_path <model_path> --num_subanchors {num_subanchors}` first."
            )
        legacy = os.path.join(anchor_path, f"{dataset_name.lower()}_emo.pt")
        anchors = torch.load(legacy, map_location="cpu")
    if anchors.dim() == 2:
        anchors = anchors.unsqueeze(1)
    return anchors
