from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch


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
        # match_score = torch.sigmoid(match).view(b,-1)
        return match  # (loss), logits, (hidden_states), (attentions)

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