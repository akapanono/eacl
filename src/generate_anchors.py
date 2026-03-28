import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import argparse
import warnings

from utils.data_process import *
from model.anchor_utils import expand_templates, get_anchor_filename, get_dataset_emotions
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='princeton-nlp/sup-simcse-roberta-large')
    parser.add_argument('--num_subanchors', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=True)

    model = AutoModel.from_pretrained(args.bert_path, local_files_only=True)
    model.eval()
    save_path = args.bert_path.split("/")[-1]
    feature_extractor = pipeline("feature-extraction",framework="pt",model=args.bert_path)
    os.makedirs(f"./emo_anchors/{save_path}", exist_ok=True)

    for dataset_name in ["IEMOCAP", "MELD", "EmoryNLP"]:
        templates = expand_templates(dataset_name, args.num_subanchors)
        emotions = get_dataset_emotions(dataset_name)
        anchors = []
        centers = []
        with torch.no_grad():
            for emotion in emotions:
                subanchor_embeddings = []
                for template in templates[emotion]:
                    emb = torch.tensor(feature_extractor(template, return_tensors="pt")[0]).mean(0)
                    subanchor_embeddings.append(emb.unsqueeze(0))
                stacked = torch.cat(subanchor_embeddings, dim=0)
                anchors.append(stacked.unsqueeze(0))
                centers.append(stacked.mean(0, keepdim=True))
        anchors = torch.cat(anchors, dim=0)
        centers = torch.cat(centers, dim=0)
        torch.save(anchors, f"./emo_anchors/{save_path}/{get_anchor_filename(dataset_name, args.num_subanchors)}")
        torch.save(centers, f"./emo_anchors/{save_path}/{dataset_name.lower()}_emo.pt")
