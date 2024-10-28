import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modelscope import AutoModelForCausalLM
from transformers import BertModel, BertTokenizer

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)
        return time_day


    
class PFA(nn.Module):
    def __init__(self, device="cuda:0", bert_layers=9):
        super(PFA, self).__init__()
        self.bert = BertModel.from_pretrained(
            './bert', output_attentions=True, output_hidden_states=True
        )
        
        self.bert.encoder.layer = self.bert.encoder.layer[:bert_layers]
        
        for layer_index, layer in enumerate(self.bert.encoder.layer):
            for name, param in layer.named_parameters():
                if layer_index < bert_layers - 5:
                    if "attention" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        
    def forward(self, x):
        return self.bert(inputs_embeds=x).last_hidden_state

    
class LLM_TFP(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=6,
        output_len=6,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len

        if num_nodes == 207 or num_nodes == 325 :
            time = 288
            
        bert_channel = 256
        to_bert_channel = 768
        
        self.Temb = TemporalEmbedding(time, bert_channel)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, bert_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            24, bert_channel, kernel_size=(1, 1)
        )
        # embedding layer

        self.bert = PFA(device=self.device, bert_layers=6)
       
        self.feature_fusion = nn.Conv2d(
            bert_channel * 3, to_bert_channel, kernel_size=(1, 1)
        )
        
        # regression
        self.regression_layer = nn.Conv2d(
            bert_channel * 3, self.output_len, kernel_size=(1, 1)
        )


    def forward(self, history_data):
        input_data = history_data
        
        batch_size, _, num_nodes, _ = input_data.shape
        history_data = history_data.permute(0, 3, 2, 1)
        
        tem_emb = self.Temb(history_data)
        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )

        input_data = self.start_conv(input_data)

        data_st = torch.cat(
            [input_data] + [tem_emb] + node_emb, dim=1
        )

        data_st = self.feature_fusion(data_st)
        
        
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)

        
        data_st = self.bert(data_st)

        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)
        
        prediction = self.regression_layer(data_st)
        return prediction