import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math





class DualCrossAttentionLayer(nn.Module):
    def __init__(self,emb_dim,num_heads,dropout):
        super(DualCrossAttentionLayer, self).__init__()
        self.attention1 = nn.MultiheadAttention(emb_dim,num_heads,dropout=dropout,batch_first=True)
        self.attention2 = nn.MultiheadAttention(emb_dim,num_heads,dropout=dropout,batch_first=True)

    def forward(self,input1,input2,mask1,mask2):
        """
        :param input1: shape (batch_size, seq_len, emb_dim)
        :param input2: shape (batch_size,seq_len, emb_dim)
        :return: (batch_size, seq_len, emb_dim) ,(batch_size, seq_len, emb_dim)
        """
        return (self.attention1(query = input1,key = input2,value = input2,key_padding_mask=~mask2.bool())[0],
                self.attention2(query = input2,key = input1,value = input1,key_padding_mask=~mask1.bool())[0])


class DualCrossAttentionFusion(nn.Module):
    def __init__(self,emb_dim,num_heads,dropout,num_layers):
        super(DualCrossAttentionFusion, self).__init__()
        self.dualCrossAttention = nn.ModuleList([
            DualCrossAttentionLayer(emb_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self,input1,input2,mask1,mask2):
        for blk in self.dualCrossAttention:
            input1,input2 = blk(input1,input2,mask1,mask2)

        return torch.cat([input1,input2],dim=1),torch.cat([mask1,mask2],dim=1)

class AttentionPooling(nn.Module):

    def __init__(self,emb_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(emb_dim,1)


    def forward(self, input_feature, attention_mask=None):
        """
        :param input_feature: shape (batch_size, seq_len, emb_dim)
        :param attention_mask: shape (batch_size, seq_len)
        :return: pooling_feature: shape (batch_size, emb_dim)
        """

        attention_scores = self.attention(input_feature).squeeze(-1) # shape (batch_size, seq_len)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1) # shape (batch_size, seq_len)
        attention_pooling_feature = torch.bmm(attention_weights.unsqueeze(1), input_feature).squeeze(1) # shape (batch_size, emb_dim)
        return attention_pooling_feature

class AvgPooling(nn.Module):
    def forward(self, input_feature, attention_mask=None):
        """
        :param input_feature: shape (batch_size, seq_len, emb_dim)
        :param attention_mask: shape (batch_size, seq_len)
        :return: pooling_feature: shape (batch_size, emb_dim)
        """
        pass



class SigmoidWithLearnableBeta(nn.Module):
    def __init__(self, init_beta=1.0):
        super(SigmoidWithLearnableBeta, self).__init__()
        # 使用nn.Parameter定义可学习的参数beta，并初始化
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        # 应用带有可学习beta的Sigmoid函数
        return torch.sigmoid(self.beta * x)

class ImageCaptionGate(nn.Module):
    def __init__(self):
        super(ImageCaptionGate, self).__init__()


    def forward(self, content_pooling_feature, caption_pooling_feature):
        """
        :param content_pooling_feature: shape (batch_size, emb_dim)
        :param caption_pooling_feature: shape (batch_size, emb_dim)
        :return: similarity: shape (batch_size)
        """
        similarity = nn.functional.cosine_similarity(content_pooling_feature, caption_pooling_feature, dim=1)
        return similarity


class ImageCaptionGate2(nn.Module):
    def __init__(self,config):
        super(ImageCaptionGate2, self).__init__()
        self.attention = nn.MultiheadAttention(config['emb_dim'],1,batch_first=True)
        self.attention_pooling = AttentionPooling(config['emb_dim'])
        self.image_classifier = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['model']['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, caption_feature,content_feature,caption_mask,content_mask):
        """
        :param caption_feature: shape (batch_size, seq_len, emb_dim)
        :param content_feature: shape (batch_size, seq_len, emb_dim)
        :param caption_mask: shape (batch_size, seq_len)
        :param content_mask: shape (batch_size, seq_len)
        :return: shape: (batch_size, 1)
        """
        content2caption_attn = self.attention(query = content_feature,
                                              key = caption_feature,
                                              value = caption_feature,
                                              key_padding_mask=~caption_mask.bool())[0]
        content2caption_attn_pooling = self.attention_pooling(content2caption_attn,attention_mask=content_mask)
        return self.image_classifier(content2caption_attn_pooling)


class FeatureAggregation(nn.Module):

    def __init__(self,emb_dim):
        super(FeatureAggregation, self).__init__()
        self.imageGate = ImageCaptionGate()
        self.maskAttention = MaskAttention(emb_dim)

    def forward(self, content_pooling_feature, caption_pooling_feature,image_pooling_feature,FTR2_pooling_feature,FTR3_pooling_feature):
        image_gate_value = self.imageGate(content_pooling_feature,caption_pooling_feature).unsqueeze(-1) # shape (batch_size,1)
        image_pooling_feature = image_gate_value * image_pooling_feature
        final_feature = torch.cat( [content_pooling_feature.unsqueeze(1),image_pooling_feature.unsqueeze(1),FTR2_pooling_feature.unsqueeze(1),FTR3_pooling_feature.unsqueeze(1)],dim=1)
        return self.maskAttention(final_feature)[0]


class FeatureNoGateAggregation(nn.Module):

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskAttention = MaskAttention(emb_dim)

    def forward(self, content_pooling_feature, image_pooling_feature, FTR2_pooling_feature,
                FTR3_pooling_feature):
        final_feature = torch.cat([content_pooling_feature.unsqueeze(1), image_pooling_feature.unsqueeze(1),
                                   FTR2_pooling_feature.unsqueeze(1), FTR3_pooling_feature.unsqueeze(1)], dim=1)
        return self.maskAttention(final_feature)[0]


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            #layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs, scores

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x shape after self attention: {}'.format(x.shape))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
    def forward(self, inputs, query, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=inputs,
                                 key=inputs,
                                 mask=mask
                                 )
        return feature, attn

def masked_softmax(scores, mask):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""

    # Fill pad positions with -inf
    scores = scores.masked_fill(mask == 0, -np.inf)
 
    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, mask_in=False):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.mask_in = mask_in
        # self.src_length_masking = src_length_masking
 
        # [hid_dim, hid_dim]
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, 1]
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        # [co_dim, 1]
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, V_mask=None, Q_mask=None):
        """ ori_setting
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        """ new_setting
        :param V: news content, batch_size * hidden_dim * content_length , eg B x 768 x 170
        :param Q: FTR info, batch_size * FTR_length * hidden_dim, eg B x 512 x 768
        :param batch_size: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """

        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
 
        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        if self.mask_in:
            # # (batch_size, 1, region_num)
            masked_a_v = masked_softmax(
                a_v.squeeze(1), V_mask
            ).unsqueeze(1)
    
            # # (batch_size, 1, seq_len)
            masked_a_q = masked_softmax(
                a_q.squeeze(1), Q_mask
            ).unsqueeze(1)
 
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(masked_a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(masked_a_q, Q))
    
            return masked_a_v, masked_a_q, v, q
        else:
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(a_q, Q))
    
            return a_v, a_q, v, q