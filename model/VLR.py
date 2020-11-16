import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import os
from transformers import BertPreTrainedModel,BertModel

class Encoder(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet121.features(x)
        x = F.relu(x)
        pred = F.adaptive_avg_pool2d(x, (1, 1))
        pred = torch.flatten(pred, 1)
        pred = self.densenet121.classifier(pred)
        return x, pred
class BertForSequenceMatch(BertPreTrainedModel):
    def __init__(self, config, alpha = 10):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.embed_layer = nn.Linear(config.hidden_size,config.hidden_size,bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        query = self.get_embedding(input_ids=input_ids, attention_mask=attention_mask)

        # match_score = torch.sigmoid(match).view(b,-1)
        return query  # (loss), logits, (hidden_states), (attentions)

    def get_embedding(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        embedding = outputs[1]
        return embedding

class Attention(nn.Module):

    def __init__(self, k_size, v_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine_v = nn.Linear(v_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k, v):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class SelfAttention(nn.Module):

    def __init__(self, k_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        content_k = self.affine_k(k)
        z = self.affine(torch.tanh(content_k)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (k * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class HAEncoder(nn.Module):
    def __init__(self, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_attn = SelfAttention(k_size=hidden_size)
        self.word_attn = SelfAttention(k_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.sent_lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.word_lstm = nn.LSTMCell(embed_size, hidden_size)

    def forward(self, captions, update_masks):
        batch_size = captions.size(0)
        num_sents = captions.size(1)
        seq_len = captions.size(2)
        embeddings = self.embed(captions)
        sent_h = torch.zeros((batch_size, self.hidden_size)).cuda()
        sent_c = torch.zeros((batch_size, self.hidden_size)).cuda()

        word_h = torch.zeros((batch_size, self.hidden_size)).cuda()
        word_c = torch.zeros((batch_size, self.hidden_size)).cuda()
        sent_hs = torch.zeros((batch_size, num_sents, self.hidden_size)).cuda()
        for k in range(num_sents):
            seq_len_k = int(update_masks[:, k].sum(dim=1).max().item())
            word_hs = torch.zeros((batch_size, seq_len_k, self.hidden_size)).cuda()
            for t in range(seq_len_k):
                batch_mask = update_masks[:, k, t].bool()
                word_h, word_c = self.word_lstm(embeddings[batch_mask, k, t], (word_h[batch_mask], word_c[batch_mask]))
                word_hs[batch_mask, t] = word_h

            sent_context = self.word_attn(word_hs)

            sent_h, sent_c = self.sent_lstm(sent_context, (sent_h, sent_c))
            sent_hs[:, k] = sent_h

        report_context = self.sent_attn(sent_hs)
        return report_context


class ReportRetrival(nn.Module):

    def __init__(self,
                 num_classes,
                 k = 5,
                 feat_size=1024,
                 text_size=768,
                 embed_size=256,
                 hidden_size=512):
        super().__init__()
        self.k = k
        self.feat_size = feat_size
        self.img_classifier = Encoder(num_classes)
        self.text_encoder = BertForSequenceMatch.from_pretrained('bert-base-uncased')
        self.atten = Attention(num_classes, feat_size)
        self.embed_layer = nn.Linear(feat_size*2, text_size, bias=False)
    def forward(self, images1, images2, captions=None, update_masks=None):
        batch_size = images1.size(0)
        pred, cnn_feats1, cnn_feats2 = self.img_classifier(images1, images2)
        cnn_feats1_t = cnn_feats1.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        cnn_feats2_t = cnn_feats2.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        context1, alpha1 = self.atten(pred, cnn_feats1_t)
        context2, alpha2 = self.atten(pred, cnn_feats2_t)
        cnn_feats = torch.cat([context1, context2], dim=1)
        cnn_feats = self.embed_layer(cnn_feats)
        if captions is not None:

            text_feats = self.text_encoder(captions, update_masks)
            cnn_feats = F.normalize(cnn_feats, dim=1, p=2)
            text_feats = F.normalize(text_feats, dim=1, p=2)
            b, c = text_feats.shape
            match = torch.bmm(cnn_feats.view(b, 1, c), text_feats.view(b, c, 1))
            match = match.view(b, -1)
            return match, pred
        else:
            base_device = self.embeddings.device
            target_device = cnn_feats.device
            query = F.normalize(cnn_feats, p=2, dim=1).to(base_device)

            match = torch.matmul(self.embeddings, query.transpose(1, 0))
            match_index = torch.argsort(match, dim=0)
            return self.dataloader.dataset.input[match_index[0]].to(target_device), pred
    def decode(self,seq, vocab):

        decoded_dict = {}
        for idx, aseq in enumerate(seq):
            decoded_dict[idx] = ['']
            pred = aseq.detach().cpu()

            words = []
            for wid in pred.tolist():
                w = vocab.idx2word[wid]
                if w == '<start>' or w == '<pad>':
                    continue
                if w == '<end>':
                    break

                words.append(w)
                if w == '.':
                    break
            decoded_dict[idx][0] += ' '.join(words)
            decoded_dict[idx][0] += ' '
        return decoded_dict


    def get_embedding_tensor(self, dataloader, device):
        self.eval()
        self.dataloader = dataloader
        embeddings = []
        with torch.no_grad():
            for index, (input, mask) in enumerate(tqdm(dataloader)):
                input, mask = input.to(device), mask.to(device)
                embedding = self.text_encoder(input_ids=input, attention_mask=mask)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        self.embeddings = embeddings




