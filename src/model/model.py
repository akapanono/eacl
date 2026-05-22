import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from model.anchor_utils import load_anchor_tensor, get_dataset_emotions


def _flag(args, name, default=False):
    return bool(getattr(args, name, default))

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
            nn.Linear(self.dim, self.num_classes)
        )
        self.use_neutral_decoupling = _flag(args, "use_neutral_decoupling")
        self.use_speaker_state = _flag(args, "use_speaker_state")
        self.use_state_fusion = _flag(args, "use_state_fusion", True)
        self.use_state_in_domain_gate = _flag(args, "use_state_in_domain_gate", True)
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)
        self.neutral_classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
        ).to(self.device)
        self.state_proj = nn.Linear(self.dim, self.dim).to(self.device)
        self.state_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        ).to(self.device)
        self.state_fusion_norm = nn.LayerNorm(self.dim).to(self.device)
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim, args.mapping_lower_dim),
            )
            for _ in range(args.num_subanchors)
        ]).to(self.device)
        gate_input_dim = self.dim
        if self.use_speaker_state and self.use_state_in_domain_gate:
            gate_input_dim += self.dim
        if self.use_neutral_decoupling:
            gate_input_dim += 1
        self.domain_gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, args.num_subanchors),
        ).to(self.device)

        self.tokenizer = tokenizer
        anchor_tensor = load_anchor_tensor(args.anchor_path, args.dataset_name, args.num_subanchors).float()
        self.register_buffer("emo_anchor", anchor_tensor.to(self.device))
        self.num_subanchors = self.emo_anchor.shape[1]
        label_names = get_dataset_emotions(args.dataset_name)
        self.label_names = label_names
        self.neutral_id = label_names.index("neutral") if "neutral" in label_names else None
        if self.neutral_id is None:
            self.use_neutral_decoupling = False
            self.args.use_neutral_decoupling = False
        self.non_neutral_label_ids = [
            idx for idx in range(self.num_classes)
            if idx != self.neutral_id
        ]
        original_to_non_neutral = torch.full((self.num_classes,), -1, dtype=torch.long)
        for non_neutral_id, original_id in enumerate(self.non_neutral_label_ids):
            original_to_non_neutral[original_id] = non_neutral_id
        self.register_buffer("original_to_non_neutral", original_to_non_neutral.to(self.device))
        self.register_buffer(
            "non_neutral_to_original",
            torch.tensor(self.non_neutral_label_ids, dtype=torch.long).to(self.device)
        )
        active_classes = len(self.non_neutral_label_ids) if self.use_neutral_decoupling else self.num_classes
        self.register_buffer(
            "emo_label",
            torch.arange(active_classes, dtype=torch.long).repeat_interleave(self.num_subanchors).to(self.device)
        )
        self.last_forward_output = {}

    def device(self):
        return self.f_context_encoder.device
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps

    def aggregate_subanchors(self, scores):
        if scores.dim() != 3:
            return scores
        if self.args.prototype_pooling == "entropy":
            domain_logits = scores.transpose(1, 2) / self.args.temp
            domain_probs = F.softmax(domain_logits, dim=-1)
            entropy = -(domain_probs * torch.log(domain_probs + self.eps)).sum(dim=-1)
            domain_weights = 1.0 / (entropy + self.args.domain_entropy_eps)
            domain_weights = domain_weights / (domain_weights.sum(dim=-1, keepdim=True) + self.eps)
            fused_probs = (domain_weights.unsqueeze(-1) * domain_probs).sum(dim=1)
            self.last_domain_probs = domain_probs.detach()
            self.last_domain_weights = domain_weights.detach()
            self.last_domain_entropy = entropy.detach()
            return torch.log(fused_probs + self.eps)
        if self.args.prototype_pooling == "logsumexp":
            return torch.logsumexp(scores / self.args.temp, dim=-1)
        return scores.max(dim=-1)[0]

    def _active_anchors(self):
        if self.use_neutral_decoupling:
            return self.emo_anchor[self.non_neutral_to_original]
        return self.emo_anchor

    def get_mapped_anchors(self):
        anchors = self._active_anchors()
        flat_anchor = anchors.reshape(-1, self.dim)
        mapped = self.map_function(flat_anchor)
        return mapped.view(anchors.shape[0], self.num_subanchors, -1)

    def get_domain_mapped_anchors(self):
        domain_anchors = []
        anchors = self._active_anchors()
        for domain_id, adapter in enumerate(self.domain_adapters):
            domain_anchor = anchors[:, domain_id, :]
            domain_anchors.append(adapter(domain_anchor).unsqueeze(1))
        return torch.cat(domain_anchors, dim=1)

    def encode_speaker_state(self, state_input_ids=None, state_attention_mask=None):
        if not self.use_speaker_state or state_input_ids is None:
            return None
        if state_attention_mask is None:
            state_attention_mask = (state_input_ids != self.pad_value).long()
        state_encoded = self.f_context_encoder(
            input_ids=state_input_ids,
            attention_mask=state_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )["last_hidden_state"]
        if getattr(self.args, "speaker_state_pooling", "mean") == "cls":
            return state_encoded[:, 0]
        mask = state_attention_mask.unsqueeze(-1).float()
        return (state_encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def fuse_speaker_state(self, mask_outputs, state_outputs):
        if state_outputs is None or not self.use_state_fusion:
            return mask_outputs
        state_projected = self.state_proj(state_outputs)
        state_alpha = self.state_gate(torch.cat([mask_outputs, state_projected], dim=-1))
        return self.state_fusion_norm(mask_outputs + state_alpha * state_projected)

    def _domain_gate_input(self, fused_outputs, state_outputs=None, neutral_prob=None):
        gate_inputs = [fused_outputs]
        if self.use_speaker_state and self.use_state_in_domain_gate:
            if state_outputs is None:
                state_outputs = torch.zeros_like(fused_outputs)
            gate_inputs.append(state_outputs)
        if self.use_neutral_decoupling:
            if neutral_prob is None:
                neutral_prob = torch.zeros(fused_outputs.shape[0], 1, device=fused_outputs.device, dtype=fused_outputs.dtype)
            gate_inputs.append(neutral_prob)
        return torch.cat(gate_inputs, dim=-1)

    def domain_gated_scores(self, mask_outputs, state_outputs=None, neutral_prob=None):
        anchors = self.get_domain_mapped_anchors()
        domain_features = []
        for adapter in self.domain_adapters:
            domain_features.append(adapter(mask_outputs).unsqueeze(1))
        domain_features = torch.cat(domain_features, dim=1)
        domain_scores = self.score_func(
            domain_features.unsqueeze(2),
            anchors.transpose(0, 1).unsqueeze(0)
        )
        domain_logits = domain_scores / self.args.temp
        domain_probs = F.softmax(domain_logits, dim=-1)
        gate_input = self._domain_gate_input(mask_outputs, state_outputs, neutral_prob)
        domain_weights = F.softmax(self.domain_gate(gate_input), dim=-1)
        fused_probs = (domain_weights.unsqueeze(-1) * domain_probs).sum(dim=1)
        self.last_emo_anchor = anchors
        self.last_domain_probs = domain_probs.detach()
        self.last_domain_weights = domain_weights.detach()
        return torch.log(fused_probs + self.eps), domain_features.mean(dim=1), anchors

    def _expand_non_neutral_scores(self, non_neutral_logits, neutral_prob):
        p_emo = F.softmax(non_neutral_logits, dim=-1)
        final_probs = torch.zeros(
            non_neutral_logits.shape[0],
            self.num_classes,
            device=non_neutral_logits.device,
            dtype=non_neutral_logits.dtype,
        )
        final_probs[:, self.neutral_id] = neutral_prob.squeeze(-1)
        final_probs[:, self.non_neutral_to_original] = (1.0 - neutral_prob) * p_emo
        return torch.log(final_probs + self.eps), final_probs, p_emo

    @torch.no_grad()
    def update_anchors(self, raw_outputs, labels):
        if self.args.disable_anchor_updates:
            return
        valid_mask = labels >= 0
        if valid_mask.sum().item() == 0:
            return

        raw_outputs = raw_outputs[valid_mask].detach()
        labels = labels[valid_mask].detach()
        if self.use_neutral_decoupling:
            non_neutral_mask = labels != self.neutral_id
            if non_neutral_mask.sum().item() == 0:
                return
            raw_outputs = raw_outputs[non_neutral_mask]
            labels = labels[non_neutral_mask]
        mapped_outputs = self.map_function(raw_outputs).detach()
        mapped_anchors = self.get_mapped_anchors().detach()

        for class_id in labels.unique().tolist():
            class_mask = labels == class_id
            class_raw = raw_outputs[class_mask]
            class_mapped = mapped_outputs[class_mask]
            if class_raw.shape[0] == 0:
                continue
            anchor_class_id = self.original_to_non_neutral[class_id].item() if self.use_neutral_decoupling else class_id
            if anchor_class_id < 0:
                continue
            sims = self.score_func(class_mapped.unsqueeze(1), mapped_anchors[anchor_class_id].unsqueeze(0))
            assignments = sims.argmax(dim=-1)
            for subanchor_id in range(self.num_subanchors):
                member_mask = assignments == subanchor_id
                if member_mask.sum().item() == 0:
                    continue
                centroid = class_raw[member_mask].mean(dim=0)
                self.emo_anchor[class_id, subanchor_id].mul_(self.args.prototype_momentum).add_(
                    centroid * (1.0 - self.args.prototype_momentum)
                )
    
    def _forward(self, sentences, state_input_ids=None, state_attention_mask=None):
        mask = 1 - (sentences == (self.pad_value)).long()

        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos]
        state_outputs = self.encode_speaker_state(state_input_ids, state_attention_mask)
        fused_outputs = self.fuse_speaker_state(mask_outputs, state_outputs)
        neutral_logit = self.neutral_classifier(fused_outputs).squeeze(-1)
        neutral_prob = torch.sigmoid(neutral_logit).unsqueeze(-1)
        self.last_forward_output = {
            "neutral_logit": neutral_logit,
            "neutral_prob": neutral_prob,
            "state_outputs": state_outputs,
            "fused_outputs": fused_outputs,
        }
        if self.use_neutral_decoupling:
            if self.args.prototype_pooling == "domain_gated":
                non_neutral_logits, mask_mapped_outputs, _ = self.domain_gated_scores(
                    fused_outputs,
                    state_outputs=state_outputs,
                    neutral_prob=neutral_prob,
                )
            else:
                mask_mapped_outputs = self.map_function(fused_outputs)
                anchors = self.get_mapped_anchors()
                self.last_emo_anchor = anchors
                subanchor_scores = self.score_func(
                    mask_mapped_outputs.unsqueeze(1).unsqueeze(2),
                    anchors.unsqueeze(0)
                )
                non_neutral_logits = self.aggregate_subanchors(subanchor_scores)
            feature, final_probs, non_neutral_probs = self._expand_non_neutral_scores(non_neutral_logits, neutral_prob)
            anchor_scores = feature if self.args.use_nearest_neighbour else None
            self.last_forward_output.update({
                "logits": feature,
                "probs": final_probs,
                "non_neutral_logits": non_neutral_logits,
                "non_neutral_probs": non_neutral_probs,
                "mask_mapped_outputs": mask_mapped_outputs,
                "raw_outputs": mask_outputs,
                "anchor_scores": anchor_scores,
            })
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        if self.args.prototype_pooling == "domain_gated":
            feature, mask_mapped_outputs, _ = self.domain_gated_scores(
                fused_outputs,
                state_outputs=state_outputs,
                neutral_prob=neutral_prob if self.use_neutral_decoupling else None,
            )
            anchor_scores = feature if self.args.use_nearest_neighbour else None
            self.last_forward_output.update({
                "logits": feature,
                "mask_mapped_outputs": mask_mapped_outputs,
                "raw_outputs": mask_outputs,
                "anchor_scores": anchor_scores,
            })
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        mask_mapped_outputs = self.map_function(fused_outputs)
        feature = torch.dropout(fused_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)
        if self.args.use_nearest_neighbour:
            anchors = self.get_mapped_anchors()
            self.last_emo_anchor = anchors
            subanchor_scores = self.score_func(
                mask_mapped_outputs.unsqueeze(1).unsqueeze(2),
                anchors.unsqueeze(0)
            )
            anchor_scores = self.aggregate_subanchors(subanchor_scores)
            if self.args.prototype_pooling == "entropy":
                feature = anchor_scores
            
        else:
            anchor_scores = None
        self.last_forward_output.update({
            "logits": feature,
            "mask_mapped_outputs": mask_mapped_outputs,
            "raw_outputs": mask_outputs,
            "anchor_scores": anchor_scores,
        })
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    
    def forward(self, sentences, state_input_ids=None, state_attention_mask=None, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(
            sentences,
            state_input_ids=state_input_ids,
            state_attention_mask=state_attention_mask,
        )
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
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
        if self.args.prototype_pooling == "domain_gated":
            domain_logits = scores.transpose(1, 2) / self.args.temp
            domain_probs = F.softmax(domain_logits, dim=-1)
            domain_weights = torch.ones(
                scores.shape[0],
                scores.shape[2],
                device=scores.device,
                dtype=scores.dtype
            ) / scores.shape[2]
            fused_probs = (domain_weights.unsqueeze(-1) * domain_probs).sum(dim=1)
            return torch.log(fused_probs + 1e-8)
        if self.args.prototype_pooling == "entropy":
            domain_logits = scores.transpose(1, 2) / self.args.temp
            domain_probs = F.softmax(domain_logits, dim=-1)
            entropy = -(domain_probs * torch.log(domain_probs + 1e-8)).sum(dim=-1)
            domain_weights = 1.0 / (entropy + self.args.domain_entropy_eps)
            domain_weights = domain_weights / (domain_weights.sum(dim=-1, keepdim=True) + 1e-8)
            fused_probs = (domain_weights.unsqueeze(-1) * domain_probs).sum(dim=1)
            return torch.log(fused_probs + 1e-8)
        if self.args.prototype_pooling == "logsumexp":
            return torch.logsumexp(scores / self.args.temp, dim=-1)
        return scores.max(dim=-1)[0]
    
    def forward(self, emb):
        scores = self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1).unsqueeze(2))
        output = self.aggregate_subanchors(scores)
        if self.args.prototype_pooling in ["entropy", "domain_gated"]:
            return output
        return output / self.args.temp
