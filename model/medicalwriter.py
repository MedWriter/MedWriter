import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import os
import numpy as np
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
        # TODO other ways of attention?
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Linear([d_model, d_k * n_heads])
        self.w_k = nn.Linear([d_model, d_k * n_heads])
        self.w_v = nn.Linear([d_model, d_v * n_heads])

        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)

        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.dropout(self.proj(context))
        return attn

class HRSentSAT(nn.Module):

    def __init__(self,
                 vocab_size,
                 num_class,
                 answer_size=768,
                 feat_size=1024,
                 embed_size=256,
                 hidden_size=512,
                 k=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.answer_size = answer_size
        self.llrm = BertForSequenceMatch.from_pretrained('bert-base-uncased')
        self.lang_atten = ScaledDotProductAttention(hidden_size)

        self.img_atten = Attention(hidden_size, feat_size)
        self.k = k
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.init_sent_h = nn.Linear(3 * feat_size, hidden_size)
        self.init_sent_c = nn.Linear(3 * feat_size, hidden_size)

        self.sent_trans = nn.Sequential(nn.Linear(answer_size, feat_size),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(feat_size, feat_size),
                                      nn.LeakyReLU(0.2)
                                      )
        self.report_trans = nn.Sequential(nn.Linear(answer_size, feat_size),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(feat_size, feat_size),
                                        nn.LeakyReLU(0.2)
                                        )
        self.keyword_fc = nn.Linear(embed_size, hidden_size)
        self.sent_lstm = nn.LSTMCell(4 * feat_size, hidden_size)
        self.word_lstm = nn.LSTMCell(embed_size + hidden_size + 4 * feat_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def set_keyword(self, vocab, keywords):
        keyword_list = [vocab(w) for w in keywords]
        self.keyword = torch.tensor(keyword_list)

    def forward(self, pred, cnn_feats1, cnn_feats2, cnn_feat_attn1, cnn_feat_attn2, report_tensor, report_embedding, captions=None, update_masks=None, stop_id=None, max_sents=10, max_len=30):
        batch_size = cnn_feats1.size(0)
        if captions is not None:
            num_sents = captions.size(1)
            seq_len = captions.size(2)
        else:
            num_sents = max_sents
            seq_len = max_len


        cnn_feats1_t = cnn_feats1.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        cnn_feats2_t = cnn_feats2.view(batch_size, self.feat_size, -1).permute(0, 2, 1)


        sent_embedding = torch.zeros((batch_size, self.k, self.feat_size)).cuda()
        init_state = torch.cat((cnn_feat_attn1, cnn_feat_attn2), dim=1)
        sent_h = self.init_sent_h(init_state)
        sent_c = self.init_sent_c(init_state)

        word_h = torch.zeros((batch_size, self.hidden_size)).cuda()
        word_c = torch.zeros((batch_size, self.hidden_size)).cuda()

        keyword_embedding = self.embed(self.keyword)
        keyword_embedding = self.keyword_fc(keyword_embedding)
        self.lang_atten(q, k, v)
        if captions is not None:
            # Teacher forcing

            logits = torch.zeros((batch_size, num_sents, seq_len, self.vocab_size)).cuda()
            embeddings = self.embed(captions)
            for k in range(num_sents):
                context1, alpha1 = self.img_atten(sent_h, cnn_feats1_t)
                context2, alpha2 = self.img_atten(sent_h, cnn_feats2_t)

                context_sent, alpha_sent = self.sent_atten(sent_h, sent_embedding)

                context = torch.cat((context1, context2, global_report_embedding, context_sent), dim=1)
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))
                seq_len_k = int(update_masks[:, k].sum(dim=1).max().item())

                for t in range(seq_len_k):
                    batch_mask = update_masks[:, k, t].bool()
                    wcontext1, walpha1 = self.img_atten(word_h, cnn_feats1_t)
                    wcontext2, walpha2 = self.img_atten(word_h, cnn_feats2_t)
                    wcontext = torch.cat((wcontext1, wcontext2), dim=1)
                    word_h_, word_c_ = self.word_lstm(
                        torch.cat((embeddings[batch_mask, k, t], sent_h[batch_mask], context[batch_mask], wcontext[batch_mask]), dim=1),
                        (word_h[batch_mask], word_c[batch_mask]))
                    indices = [*batch_mask.unsqueeze(1).repeat(1, self.hidden_size).nonzero().t()]
                    word_h = word_h.index_put(indices, word_h_.view(-1))
                    word_c = word_c.index_put(indices, word_c_.view(-1))
                    logits[batch_mask, k, t] = self.fc(self.dropout(word_h[batch_mask]))
                sent_answers = self.llrm.retrivel_tensor(captions[:, k], k=self.k)
                sent_embedding = self.sent_trans(sent_answers)
            return logits

        else:
            result = torch.zeros((batch_size, num_sents, seq_len), dtype=torch.long).cuda()
            x_t = cnn_feats1.new_full((batch_size,), 1, dtype=torch.long)

            for k in range(num_sents):

                context1, alpha1 = self.img_atten(sent_h, cnn_feats1_t)
                context2, alpha2 = self.img_atten(sent_h, cnn_feats2_t)

                context_sent, alpha_sent = self.sent_atten(sent_h, sent_embedding)
                context = torch.cat((context1, context2, global_report_embedding, context_sent), dim=1)
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))
                sent_len = 0
                for t in range(seq_len):
                    embedding = self.embed(x_t)
                    wcontext1, walpha1 = self.img_atten(word_h, cnn_feats1_t)
                    wcontext2, walpha2 = self.img_atten(word_h, cnn_feats2_t)
                    wcontext = torch.cat((wcontext1, wcontext2), dim=1)
                    word_h, word_c = self.word_lstm(torch.cat((embedding, sent_h, context, wcontext), dim=1),
                                                    (word_h, word_c))
                    logit = self.fc(word_h)
                    x_t = logit.argmax(dim=1)
                    result[:, k, t] = x_t
                    sent_len += 1
                    if x_t[0] == stop_id:
                        break
                sent_answers = self.llrm.retrivel_tensor(result[:, k], k=self.k)
                sent_embedding = self.sent_trans(sent_answers)

            return result

class BertForSequenceMatch(BertPreTrainedModel):
    def __init__(self, config, alpha = 10):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.embed_layer = nn.Linear(config.hidden_size,config.hidden_size,bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids1=None,
        attention_mask1=None,
        input_ids2=None,
        attention_mask2=None,
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

        query = self.get_embedding(input_ids=input_ids1, attention_mask=attention_mask1)

        answer = self.get_embedding(input_ids=input_ids2, attention_mask=attention_mask2)

        key = self.embed_layer(query)
        answer = F.normalize(answer, dim=1, p=2)
        key = F.normalize(key, dim=1, p=2)
        b,c = key.shape
        match = torch.bmm(answer.view(b,1,c),key.view(b,c,1))
        match = match.view(b,-1)

        return match

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
        embeddings = []
        with torch.no_grad():
            for index, (input, mask) in enumerate(tqdm(dataloader)):
                input, mask = input.to(device), mask.to(device)
                embedding = self.get_embedding(input, attention_mask=mask)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        self.embeddings = embeddings


    def retrivel_tensor(self, query, k=5):
        # query: a batch of query tensors bx sent_len
        mask = query != 0 # torch.tensor(query.clone().detach()!=0, dtype=torch.bool).to(query.device)
        base_device = self.embeddings.device
        target_device = query.device
        query = self.get_embedding(query, attention_mask=mask)
        query = self.embed_layer(query)
        query = F.normalize(query, p=2, dim=1)
        query = query.to(base_device)
        match = torch.matmul(self.embeddings, query.transpose(1,0))
        match_index = torch.argsort(match,dim=0)
        answer_tokens = self.embeddings[match_index[:k]].permute(1,0,2)

        return answer_tokens.to(target_device)


class REncoder2Decoder_hia(nn.Module):

    def __init__(self, num_classes, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()

        self.vlrm = VLRM(num_classes)
        self.decoder = HRSentSAT(
                                num_class=num_classes,
                                vocab_size=vocab_size,
                                feat_size=feat_size,
                                embed_size=embed_size,
                                hidden_size=hidden_size)

    def forward(self, images1, images2, captions=None, update_masks=None, stop_id=None, max_sents=10, max_len=30):
        answer_tensor, answer_embedding, cnn_feats1, cnn_feats2, cnn_feat_attn1, cnn_feat_attn2, pred = self.vlrm(images1,images2)

        return self.decoder(pred, cnn_feats1, cnn_feats2, cnn_feat_attn1, cnn_feat_attn2, answer_tensor, answer_embedding, captions, update_masks, stop_id, max_sents, max_len), pred

class VLRM(nn.Module):

    def __init__(self,
                 num_classes,
                 k=5,
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
        cnn_feats1, pred1 = self.img_classifier(images1)
        cnn_feats2, pred2 = self.img_classifier(images2)
        pred = pred1 + pred2
        cnn_feats1_t = cnn_feats1.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        cnn_feats2_t = cnn_feats2.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        context1, alpha1 = self.atten(pred, cnn_feats1_t)
        context2, alpha2 = self.atten(pred, cnn_feats2_t)
        cnn_feats = torch.cat([context1, context2], dim=1)
        cnn_feats = self.embed_layer(cnn_feats)

        base_device = self.embeddings.device
        target_device = cnn_feats.device
        query = F.normalize(cnn_feats, p=2, dim=1).to(base_device)

        match = torch.matmul(self.embeddings, query.transpose(1, 0))
        match_index = torch.argsort(match, dim=0)
        return self.dataloader.dataset.input[match_index[:self.k]].to(target_device), \
               self.embeddings[match_index[:self.k]].permute(1,0,2).to(target_device), \
               cnn_feats1, cnn_feats2, context1, context2, pred

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
                embedding = self.text_encoder.get_embedding(input_ids=input, attention_mask=mask)
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        self.embeddings = embeddings


