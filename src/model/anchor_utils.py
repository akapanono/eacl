import os

import torch

DOMAIN_NAMES = [
    "valence",
    "arousal",
    "dominance_control",
    "social_appraisal",
    "discourse_context",
]


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
                "The speaker feels neutral, neither clearly positive nor clearly negative.",
                "The speaker feels neutral with low arousal and steady emotional energy.",
                "The speaker feels neutral with balanced control and little pressure to act.",
                "The speaker takes a matter-of-fact social stance without strong approval or rejection.",
                "The speaker remains neutral because the dialogue context does not create a strong emotional shift.",
            ],
            "excited": [
                "The speaker feels excited with strongly positive valence.",
                "The speaker feels excited with very high arousal and energetic anticipation.",
                "The speaker feels excited with eager approach motivation and readiness to act.",
                "The speaker is socially engaged and responds with lively enthusiasm.",
                "The speaker becomes excited because the dialogue context is rewarding, stimulating, or surprising in a positive way.",
            ],
            "frustrated": [
                "The speaker feels frustrated with negative valence caused by blocked goals.",
                "The speaker feels frustrated with tense and rising arousal.",
                "The speaker feels reduced control because something prevents the desired outcome.",
                "The speaker reacts with dissatisfaction, resistance, or impatience toward the interaction.",
                "The speaker becomes frustrated because repeated obstacles or misunderstandings build up in the dialogue.",
            ],
            "sad": [
                "The speaker feels sad with negative valence and emotional pain.",
                "The speaker feels sad with low arousal, low energy, and heaviness.",
                "The speaker feels low control or helplessness about the situation.",
                "The speaker withdraws socially or seeks comfort and support.",
                "The speaker becomes sad because the dialogue context suggests loss, disappointment, rejection, or regret.",
            ],
            "happy": [
                "The speaker feels happy with clearly positive valence.",
                "The speaker feels happy with moderate, pleasant arousal.",
                "The speaker feels comfortable and in control of the situation.",
                "The speaker is socially warm, open, and friendly.",
                "The speaker stays happy because the dialogue context is pleasant, supportive, or successful.",
            ],
            "angry": [
                "The speaker feels angry with strong negative valence toward a person or situation.",
                "The speaker feels angry with high arousal, tension, and agitation.",
                "The speaker feels a strong need to regain control or challenge what happened.",
                "The speaker takes a confrontational, accusatory, or defensive social stance.",
                "The speaker becomes angry because the dialogue context suggests conflict, blame, unfairness, or provocation.",
            ],
        }
    if dataset_name == "MELD":
        return {
            "anger": [
                "The speaker feels anger with strong negative valence toward a person or situation.",
                "The speaker feels anger with high arousal, tension, and agitation.",
                "The speaker feels a strong need to regain control or challenge what happened.",
                "The speaker takes a confrontational, accusatory, or defensive social stance.",
                "The speaker becomes angry because the dialogue context suggests conflict, blame, unfairness, or provocation.",
            ],
            "disgust": [
                "The speaker feels disgust with negative valence and aversion.",
                "The speaker feels disgust with tense but controlled arousal.",
                "The speaker rejects the situation and wants distance rather than control.",
                "The speaker shows social disapproval, contempt, or dismissal.",
                "The speaker becomes disgusted because the dialogue context presents something offensive, inappropriate, or repellent.",
            ],
            "fear": [
                "The speaker feels fear with negative valence and perceived threat.",
                "The speaker feels fear with high alert arousal and nervous tension.",
                "The speaker feels low control and uncertainty about what may happen.",
                "The speaker responds cautiously, defensively, or with a need for reassurance.",
                "The speaker becomes fearful because the dialogue context suggests risk, danger, uncertainty, or vulnerability.",
            ],
            "joy": [
                "The speaker feels joy with clearly positive valence.",
                "The speaker feels joy with lively and pleasant arousal.",
                "The speaker feels comfortable, safe, and able to enjoy the moment.",
                "The speaker is socially warm, playful, affectionate, or connected.",
                "The speaker becomes joyful because the dialogue context is rewarding, funny, successful, or supportive.",
            ],
            "sadness": [
                "The speaker feels sadness with negative valence and emotional pain.",
                "The speaker feels sadness with low arousal, low energy, and heaviness.",
                "The speaker feels low control or helplessness about the situation.",
                "The speaker withdraws socially or seeks comfort and support.",
                "The speaker becomes sad because the dialogue context suggests loss, disappointment, rejection, or regret.",
            ],
            "surprise": [
                "The speaker feels surprise with valence that depends on an unexpected event.",
                "The speaker feels surprise with sudden high arousal and quick attention.",
                "The speaker temporarily loses prediction or control because something unexpected occurs.",
                "The speaker reacts socially with curiosity, disbelief, or a startled response.",
                "The speaker becomes surprised because the dialogue context changes unexpectedly or reveals new information.",
            ],
            "neutral": [
                "The speaker feels neutral, neither clearly positive nor clearly negative.",
                "The speaker feels neutral with low arousal and steady emotional energy.",
                "The speaker feels neutral with balanced control and little pressure to act.",
                "The speaker takes a matter-of-fact social stance without strong approval or rejection.",
                "The speaker remains neutral because the dialogue context does not create a strong emotional shift.",
            ],
        }
    if dataset_name == "EmoryNLP":
        return {
            "joyful": [
                "The speaker feels joyful with clearly positive valence.",
                "The speaker feels joyful with lively and pleasant arousal.",
                "The speaker feels comfortable, safe, and able to enjoy the moment.",
                "The speaker is socially warm, playful, affectionate, or connected.",
                "The speaker becomes joyful because the dialogue context is rewarding, funny, successful, or supportive.",
            ],
            "neutral": [
                "The speaker feels neutral, neither clearly positive nor clearly negative.",
                "The speaker feels neutral with low arousal and steady emotional energy.",
                "The speaker feels neutral with balanced control and little pressure to act.",
                "The speaker takes a matter-of-fact social stance without strong approval or rejection.",
                "The speaker remains neutral because the dialogue context does not create a strong emotional shift.",
            ],
            "powerful": [
                "The speaker feels powerful with positive valence from confidence and agency.",
                "The speaker feels powerful with controlled arousal and strong energy.",
                "The speaker feels high dominance, control, and ability to influence the situation.",
                "The speaker takes an assertive, confident, or leading social stance.",
                "The speaker becomes powerful because the dialogue context gives advantage, authority, competence, or control.",
            ],
            "mad": [
                "The speaker feels mad with strong negative valence toward a person or situation.",
                "The speaker feels mad with high arousal, tension, and agitation.",
                "The speaker feels a strong need to regain control or challenge what happened.",
                "The speaker takes a confrontational, accusatory, or defensive social stance.",
                "The speaker becomes mad because the dialogue context suggests conflict, blame, unfairness, or provocation.",
            ],
            "scared": [
                "The speaker feels scared with negative valence and perceived threat.",
                "The speaker feels scared with high alert arousal and nervous tension.",
                "The speaker feels low control and uncertainty about what may happen.",
                "The speaker responds cautiously, defensively, or with a need for reassurance.",
                "The speaker becomes scared because the dialogue context suggests risk, danger, uncertainty, or vulnerability.",
            ],
            "peaceful": [
                "The speaker feels peaceful with mildly positive or calm neutral valence.",
                "The speaker feels peaceful with very low arousal, relaxation, and low tension.",
                "The speaker feels safe, stable, and in quiet control.",
                "The speaker is socially gentle, cooperative, and non-confrontational.",
                "The speaker remains peaceful because the dialogue context is stable, safe, and non-threatening.",
            ],
            "sad": [
                "The speaker feels sad with negative valence and emotional pain.",
                "The speaker feels sad with low arousal, low energy, and heaviness.",
                "The speaker feels low control or helplessness about the situation.",
                "The speaker withdraws socially or seeks comfort and support.",
                "The speaker becomes sad because the dialogue context suggests loss, disappointment, rejection, or regret.",
            ],
        }
    raise ValueError(f"Unsupported dataset for domain templates: {dataset_name}")


def get_standard_anchor_templates(dataset_name):
    if dataset_name == "IEMOCAP":
        return {
            "neutral": [
                "The speaker's emotion is neutral: calm, steady, and emotionally balanced.",
                "The speaker is neutral and matter-of-fact, without strong positive or negative feeling.",
                "The speaker feels composed and unreactive in this utterance.",
            ],
            "excited": [
                "The speaker's emotion is excited: energetic, eager, and strongly positive.",
                "The speaker feels thrilled, enthusiastic, and highly activated.",
                "The speaker is animated and expressive with positive anticipation.",
            ],
            "frustrated": [
                "The speaker's emotion is frustrated: blocked, tense, and dissatisfied.",
                "The speaker feels impatient because the situation is not working as expected.",
                "The speaker is irritated by obstacles, limits, or repeated difficulty.",
            ],
            "sad": [
                "The speaker's emotion is sad: hurt, low in energy, and emotionally heavy.",
                "The speaker feels disappointed, discouraged, or sorrowful.",
                "The speaker is withdrawn or vulnerable because of loss, regret, or disappointment.",
            ],
            "happy": [
                "The speaker's emotion is happy: cheerful, pleased, and positive.",
                "The speaker feels warm, relaxed, and content.",
                "The speaker is lighthearted and socially open in this utterance.",
            ],
            "angry": [
                "The speaker's emotion is angry: tense, irritated, and confrontational.",
                "The speaker feels offended or provoked and wants to challenge the situation.",
                "The speaker is agitated by conflict, unfairness, or blame.",
            ],
        }
    if dataset_name == "MELD":
        return {
            "anger": [
                "The speaker's emotion is anger: tense, irritated, and confrontational.",
                "The speaker feels offended or provoked and wants to challenge the situation.",
                "The speaker is agitated by conflict, unfairness, or blame.",
            ],
            "disgust": [
                "The speaker's emotion is disgust: repelled, rejecting, and uncomfortable.",
                "The speaker feels aversion toward something offensive or inappropriate.",
                "The speaker is dismissive or contemptuous toward the situation.",
            ],
            "fear": [
                "The speaker's emotion is fear: nervous, threatened, and uncertain.",
                "The speaker feels afraid, alarmed, or vulnerable.",
                "The speaker is cautious or defensive because of risk or danger.",
            ],
            "joy": [
                "The speaker's emotion is joy: cheerful, pleased, and positive.",
                "The speaker feels delighted, playful, or warmly connected.",
                "The speaker is bright and socially open in this utterance.",
            ],
            "sadness": [
                "The speaker's emotion is sadness: hurt, low in energy, and emotionally heavy.",
                "The speaker feels disappointed, lonely, or discouraged.",
                "The speaker is withdrawn or vulnerable because of loss, regret, or rejection.",
            ],
            "surprise": [
                "The speaker's emotion is surprise: startled, reactive, and caught off guard.",
                "The speaker feels sudden attention because something unexpected happened.",
                "The speaker is amazed, curious, confused, or disbelieving.",
            ],
            "neutral": [
                "The speaker's emotion is neutral: calm, steady, and emotionally balanced.",
                "The speaker is neutral and matter-of-fact, without strong positive or negative feeling.",
                "The speaker feels composed and unreactive in this utterance.",
            ],
        }
    if dataset_name == "EmoryNLP":
        return {
            "joyful": [
                "The speaker's emotion is joyful: cheerful, pleased, and positive.",
                "The speaker feels delighted, playful, or warmly connected.",
                "The speaker is bright and socially open in this utterance.",
            ],
            "neutral": [
                "The speaker's emotion is neutral: calm, steady, and emotionally balanced.",
                "The speaker is neutral and matter-of-fact, without strong positive or negative feeling.",
                "The speaker feels composed and unreactive in this utterance.",
            ],
            "powerful": [
                "The speaker's emotion is powerful: confident, assertive, and in control.",
                "The speaker feels capable, dominant, or self-assured.",
                "The speaker is commanding or influential in this utterance.",
            ],
            "mad": [
                "The speaker's emotion is mad: tense, irritated, and confrontational.",
                "The speaker feels offended or provoked and wants to challenge the situation.",
                "The speaker is agitated by conflict, unfairness, or blame.",
            ],
            "scared": [
                "The speaker's emotion is scared: nervous, threatened, and uncertain.",
                "The speaker feels afraid, alarmed, or vulnerable.",
                "The speaker is cautious or defensive because of risk or danger.",
            ],
            "peaceful": [
                "The speaker's emotion is peaceful: calm, safe, and relaxed.",
                "The speaker feels settled, composed, and untroubled.",
                "The speaker is gentle and non-confrontational in this utterance.",
            ],
            "sad": [
                "The speaker's emotion is sad: hurt, low in energy, and emotionally heavy.",
                "The speaker feels disappointed, lonely, or discouraged.",
                "The speaker is withdrawn or vulnerable because of loss, regret, or rejection.",
            ],
        }
    raise ValueError(f"Unsupported dataset for standard templates: {dataset_name}")


def expand_templates(dataset_name, num_subanchors):
    templates = get_domain_subanchor_templates(dataset_name)
    fallback_templates = get_standard_anchor_templates(dataset_name)
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
