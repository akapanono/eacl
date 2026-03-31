import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from model.anchor_utils import load_anchor_tensor

class CLModel(nn.Module):
    def __init__(self, args, n_classes, tokenizer=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = n_classes
        self.pad_value = args.pad_value
        self.mask_value = 50265
        self.f_context_encoder = AutoModel.from_pretrained(args.bert_path, local_files_only=True)
        
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.avg_dist = []

        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        self.eps = 1e-8
        self.device = f"cuda:{self.args.gpu_id}" if self.args.cuda else "cpu"
        self.predictor = nn.Sequential(
            # nn.Linear(self.dim, self.dim),
            # nn.ReLU(),
            nn.Linear(args.mapping_lower_dim, self.num_classes)
        )
        self.semantic_projector = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)
        self.speaker_embedding = nn.Embedding(args.max_speakers, args.mapping_lower_dim).to(self.device)
        self.anchor_prior_proj = nn.Sequential(
            nn.Linear(args.mapping_lower_dim, args.mapping_lower_dim),
            nn.LayerNorm(args.mapping_lower_dim),
            nn.Tanh(),
        ).to(self.device)
        transfer_dim = args.mapping_lower_dim * 3
        self.transfer_gate = nn.Sequential(
            nn.Linear(transfer_dim, args.mapping_lower_dim),
            nn.Sigmoid(),
        ).to(self.device)
        self.transfer_fusion = nn.Sequential(
            nn.Linear(transfer_dim, args.mapping_lower_dim),
            nn.LayerNorm(args.mapping_lower_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ).to(self.device)

        self.tokenizer = tokenizer
        anchor_tensor = load_anchor_tensor(args.anchor_path, args.dataset_name, args.num_subanchors).float()
        self.register_buffer("emo_anchor", anchor_tensor.to(self.device))
        self.num_subanchors = self.emo_anchor.shape[1]
        self.register_buffer(
            "emo_label",
            torch.arange(self.num_classes, dtype=torch.long).repeat_interleave(self.num_subanchors).to(self.device)
        )

    def device(self):
        return self.f_context_encoder.device
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps

    def aggregate_subanchors(self, scores):
        if scores.dim() != 3:
            return scores
        if self.args.prototype_pooling == "logsumexp":
            return torch.logsumexp(scores / self.args.temp, dim=-1)
        return scores.max(dim=-1)[0]

    def get_mapped_anchors(self):
        flat_anchor = self.emo_anchor.view(-1, self.dim)
        mapped = self.map_function(flat_anchor)
        return mapped.view(self.num_classes, self.num_subanchors, -1)

    def build_affective_representation(self, mask_outputs, speaker_ids):
        semantic_outputs = self.semantic_projector(mask_outputs)
        anchors = self.get_mapped_anchors()
        class_anchors = anchors.mean(dim=1)
        anchor_weights = torch.softmax(
            self.score_func(
                semantic_outputs.unsqueeze(1),
                class_anchors.unsqueeze(0)
            ) / self.args.transfer_temperature,
            dim=-1
        )
        anchor_prior = torch.matmul(anchor_weights, class_anchors)
        anchor_prior = self.anchor_prior_proj(anchor_prior)
        speaker_states = self.speaker_embedding(speaker_ids)
        transfer_inputs = torch.cat([semantic_outputs, anchor_prior, speaker_states], dim=-1)
        transfer_gate = self.transfer_gate(transfer_inputs)
        fused = self.transfer_fusion(transfer_inputs)
        affective_outputs = transfer_gate * semantic_outputs + (1.0 - transfer_gate) * fused
        return semantic_outputs, affective_outputs, anchor_weights, class_anchors

    @torch.no_grad()
    def update_anchors(self, raw_outputs, labels):
        if self.args.disable_anchor_updates:
            return
        valid_mask = labels >= 0
        if valid_mask.sum().item() == 0:
            return

        raw_outputs = raw_outputs[valid_mask].detach()
        labels = labels[valid_mask].detach()
        mapped_outputs = self.map_function(raw_outputs).detach()
        mapped_anchors = self.get_mapped_anchors().detach()

        for class_id in labels.unique().tolist():
            class_mask = labels == class_id
            class_raw = raw_outputs[class_mask]
            class_mapped = mapped_outputs[class_mask]
            if class_raw.shape[0] == 0:
                continue
            sims = self.score_func(class_mapped.unsqueeze(1), mapped_anchors[class_id].unsqueeze(0))
            assignments = sims.argmax(dim=-1)
            for subanchor_id in range(self.num_subanchors):
                member_mask = assignments == subanchor_id
                if member_mask.sum().item() == 0:
                    continue
                centroid = class_raw[member_mask].mean(dim=0)
                self.emo_anchor[class_id, subanchor_id].mul_(self.args.prototype_momentum).add_(
                    centroid * (1.0 - self.args.prototype_momentum)
                )
    
    def _forward(self, sentences, speaker_ids):
        mask = 1 - (sentences == (self.pad_value)).long()

        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos]
        semantic_outputs, mask_mapped_outputs, anchor_weights, class_anchors = self.build_affective_representation(
            mask_outputs,
            speaker_ids
        )
        feature = torch.dropout(mask_mapped_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)
        if self.args.use_nearest_neighbour:
            anchors = self.get_mapped_anchors()
            self.last_emo_anchor = anchors
            subanchor_scores = self.score_func(
                mask_mapped_outputs.unsqueeze(1).unsqueeze(2),
                anchors.unsqueeze(0)
            )
            anchor_scores = self.aggregate_subanchors(subanchor_scores)
            
        else:
            anchor_scores = None
        return feature, mask_mapped_outputs, mask_outputs, semantic_outputs, anchor_weights, class_anchors, anchor_scores
    
    def forward(self, sentences, speaker_ids=None, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        if speaker_ids is None:
            speaker_ids = torch.zeros(sentences.shape[0], dtype=torch.long, device=sentences.device)
        feature, mask_mapped_outputs, mask_outputs, semantic_outputs, anchor_weights, class_anchors, anchor_scores = self._forward(sentences, speaker_ids)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, semantic_outputs, anchor_weights, class_anchors, anchor_scores
        else:
            return feature
        
class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8

    def aggregate_subanchors(self, scores):
        if self.args.prototype_pooling == "logsumexp":
            return torch.logsumexp(scores / self.args.temp, dim=-1)
        return scores.max(dim=-1)[0]
    
    def forward(self, emb):
        scores = self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1).unsqueeze(2))
        return self.aggregate_subanchors(scores) / self.args.temp
