import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import framework.configbase
import math
import time
import numpy as np
from modules.transformer_encoder import Encoder, FaceEncoder
from modules.transformer_decoder import Decoder, Ptr_Gate
from modules.common import RawMultiHeadAttention
import pdb
import json

decay1 = [(i+1)*20**(-1) for i in range(20)]
decay2 = [1-(i+1)*50**(-1) for i in range(50)]


class TransformerConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super(TransformerConfig, self).__init__()
    self.vocab = 0
    self.max_words_in_sent = 150
    self.ft_dim = 4096
    self.d_model = 512
    self.enc_n_layers = 3
    self.dec_n_layers = 3
    self.heads = 8
    self.dropout = 0.1
    self.keyframes = False
    self.rl = False
    self.document_freq = None

# itos = json.load(open('/data2/yzh/Dataset/MOVIES/annotation/811/vocab/origin/id2c.json'))

class Transformer(nn.Module):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.config = config
    self.encoder = Encoder(self.config.ft_dim, self.config.d_model, self.config.enc_n_layers, self.config.heads, self.config.dropout, self.config.keyframes)
    # self.r_encoder = RoleEncoder(self.config.d_model, self.config.enc_n_layers, self.config.heads, self.config.dropout)
    self.f_encoder = FaceEncoder(self.config.d_model, self.config.enc_n_layers, self.config.heads, self.config.dropout)
    self.decoder = Decoder(self.config.vocab, self.config.d_model, self.config.dec_n_layers, self.config.heads, self.config.dropout)
    self.dropout = nn.Dropout(self.config.dropout)
    self.logit = nn.Linear(self.config.d_model, self.config.vocab)
    self.logit.weight = self.decoder.embed.embed.weight
    self.remove_gate = nn.Linear(self.config.d_model, 1)
    self.add_gate = nn.Linear(self.config.d_model, 1)
    self.q_linear = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
    self.next_attn = nn.Linear(2*self.config.d_model, 1)
    self.init_weights()
    # self.ptr_gate = Ptr_Gate(self.config.d_model)
    self.role_selector = RawMultiHeadAttention(self.config.heads, self.config.d_model, self.config.dropout)
    self.attn_log = open('/data2/yzh/cloned_repo/video-paragraph/results/movie/dm.token/attn_log.txt', 'a')

  def init_weights(self,):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, trg, src_mask, trg_mask, roleface, rolename_mask):
    e_outputs, org_key, select = self.encoder(src, src_mask)
    # r_outputs = self.r_encoder(rolename, roleface, rolename_mask)
    f_outputs = self.f_encoder(roleface)
    d_output, attn_weights = [], []
    output = []    

    # initialize memory
    add_state = torch.tensor(decay2[:e_outputs.size(1)]+[0]*max(0,e_outputs.size(1)-50)).cuda().unsqueeze(0).unsqueeze(-1)
    memory_bank = e_outputs * add_state

    for i in range(1, trg.size(1)+1):
      # decode
      # pdb.set_trace()
      word, attn = self.decoder(trg[:,i-1].unsqueeze(1), memory_bank, src_mask, trg_mask[:,i-1,:i].unsqueeze(1), roleface, step=i)
      d_output.append(word[:,-1])
      attn_weights.append(attn[:,:,-1].mean(dim=1))
      # select frame
      # 这里别softmax，下面也是，否则会给padding分配权重
      frame_select = attn_weights[-1]
      selected_frame = torch.bmm(frame_select.unsqueeze(1), memory_bank)
      # update memory
      memory_bank, add_state = self.update_memory(memory_bank, add_state, e_outputs, attn_weights[-20:], d_output[-20:])
      # attend to role faces
      _, role_attn = self.role_selector(selected_frame, f_outputs, f_outputs)
      role_attn_weights = role_attn.mean(dim=1) # [batch, 1, seq_len]
      # comput distribution
      vocab_distribution = self.logit(word[:,-1]).unsqueeze(1) # [batch, 1, vocab]
      # vocab_distribution = torch.softmax(vocab_distribution, dim=-1).unsqueeze(1) # [batch, 1, vocab]
      # pdb.set_trace()
      # cancat distribution
      joint_distribution = torch.cat([vocab_distribution, role_attn_weights], dim=-1) # [batch, 1, vocab+10]

      output.append(joint_distribution)
    output = torch.cat([_.unsqueeze(1) for _ in output], dim=1)
    # output = self.logit(torch.cat([_.unsqueeze(1) for _ in d_output], 1)) # [batch_size, trg_len, vocab_size]
    return output, org_key, select

  def update_memory(self, memory_bank, add_state, e_outputs, attn, query_s):
    remove_prob = torch.sigmoid(self.remove_gate(query_s[-1])).unsqueeze(-1)
    add_prob = torch.sigmoid(self.add_gate(query_s[-1])).unsqueeze(-1)
    temp = torch.softmax(torch.tensor(decay1[20-len(attn):]).cuda(), dim=-1)
    attn = sum([attn[i]*temp[i] for i in range(len(attn))]).unsqueeze(-1)
    # remove for diversity
    query_s = sum([query_s[i]*temp[i] for i in range(len(query_s))])
    sim = torch.sigmoid(torch.matmul(memory_bank, self.q_linear(query_s).unsqueeze(-1)))
    memory_bank = memory_bank * (1 - remove_prob * attn * sim)
    # add for coherence
    last_ctx = (e_outputs * attn).sum(dim=1, keepdim=True)
    next_attn = torch.sigmoid(self.next_attn(torch.cat([e_outputs,last_ctx.expand_as(e_outputs)], dim=-1)))
    memory_bank = memory_bank + e_outputs * (1-add_state) * (add_prob*next_attn)
    add_state = add_state + (1-add_state) * (add_prob*next_attn)
    return memory_bank, add_state

  def sample(self, src, src_mask, rolename, roleface, rolename_mask, decoding='greedy'):
    init_tok = 2
    eos_tok = 3
    if self.config.keyframes:
      e_outputs, src_mask = self.encoder.get_keyframes(src, src_mask)
    else:
      e_outputs, _, _ = self.encoder(src, src_mask)
    # r_outputs = self.r_encoder(rolename, roleface, rolename_mask)
    f_outputs = self.f_encoder(roleface)
    # initialize memory
    add_state = torch.tensor(decay2[:e_outputs.size(1)]+[0]*max(0,e_outputs.size(1)-50)).cuda().unsqueeze(0).unsqueeze(-1)
    memory_bank = e_outputs * add_state
    outputs = torch.ones(src.size(0), 1).fill_(init_tok).long().cuda()
    seqLogprobs = torch.zeros(src.size(0), 60).cuda()
    attn_weights, d_output = [], []

    for i in range(1, 60):
      # decode
      trg_mask = self.nopeak_mask(i)
      word, attn = self.decoder(outputs[:,-1].unsqueeze(1), memory_bank, src_mask, trg_mask[:,-1].unsqueeze(1), roleface, step=i)
      attn_weights.append(attn[:,:,-1].mean(dim=1))
      d_output.append(word[:,-1])
      # select frame
      frame_select = attn_weights[-1]
      selected_frame = torch.bmm(frame_select.unsqueeze(1), memory_bank).squeeze(1)
      # attend to role tokens
      _, role_attn = self.role_selector(selected_frame, f_outputs, f_outputs)
      role_attn_weights = role_attn.mean(dim=1)
      # 追加写入attn_log.txt
      self.attn_log.write('frame:\t'+str(frame_select[0])+'\nrole:\t'+str(role_attn_weights[0])+'\n')
      # compute distribution
      vocab_distribution = self.logit(word[:,-1]).unsqueeze(1)
      joint_distribution = torch.cat([vocab_distribution, role_attn_weights], dim=-1)
      
      logprobs = F.log_softmax(joint_distribution, dim=-1)
      # pdb.set_trace()
      if decoding == 'greedy':
        _, next_word = torch.max(logprobs, dim=-1)
      else:
        probs = torch.exp(logprobs.data).cpu()
        next_word = torch.multinomial(probs, 1).cuda()
        seqLogprobs[:,i] = logprobs.gather(1, next_word).view(-1)
      outputs = torch.cat([outputs, next_word], dim=1)
      memory_bank, add_state = self.update_memory(memory_bank, add_state, e_outputs, attn_weights[-20:], d_output[-20:])
    attn_weights = torch.cat([_.unsqueeze(1) for _ in attn_weights], dim=1)
    return outputs, seqLogprobs, attn_weights

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

