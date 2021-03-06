from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import math
import torch.utils.data

UNK, PAD, BOS, EOS = 0, 1, 2, 3


class CaptionDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_root, cap_file, word2int, int2word,
    max_words_in_sent=150, is_train=False, _logger=None):
    super(CaptionDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    # self.names = np.load(name_file)
    # self.num_ft = len(self.names)
    # self.print_fn('names size %d' % self.num_ft)

    self.ref_captions = json.load(open(cap_file)) # gt
    self.names = list(self.ref_captions.keys())
    # self.train_ref = json.load(open(cap_file['trn']))
    # self.val_ref = json.load(open(cap_file['val']))
    # self.test_ref = json.load(open(cap_file['tst']))
    # self.val_tst_ref = self.val_ref
    # self.val_tst_ref.update(self.test_ref)

    # self.captions, self.cap2ftid = [], []
    # for ftid, name in enumerate(self.names):
    #   self.captions.extend(self.ref_captions[name])
    #   self.cap2ftid.extend([ftid] * len(self.ref_captions[name]))
    # self.cap2ftid = np.array(self.cap2ftid)
    # self.num_caption = len(self.captions)
    # self.print_fn('captions size %d' % self.num_caption)
    
    self.stoi = json.load(open(word2int))
    self.itos = json.load(open(int2word))
    self.ft_root = ft_root
    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.movie2id = json.load(open('/data2/yzh/Dataset/MOVIES/annotation/811/vocab/movie_vocab/c2id.json'))

  def temporal_pad_or_trim_feature(self, ft, max_len, transpose=False, average=False):
    length, dim_ft = ft.shape
    # pad
    if length <= max_len:
      ft_new = np.zeros((max_len, dim_ft), np.float32)
      ft_new[:length] = ft
    # trim
    else:
      if average:
        indices = np.round(np.linspace(0, length, max_len+1)).astype(np.int32)
        ft_new = [np.mean(ft[indices[i]: indices[i+1]], axis=0) for i in range(max_len)]
        ft_new = np.array(ft_new, np.float32)
      else:
        indices = np.round(np.linspace(0, length - 1, max_len)).astype(np.int32)
        ft_new = ft[indices]
    if transpose:
      ft_new = ft_new.transpose()
    return ft_new

  def pad_sent(self, x):
    max_len = self.max_words_in_sent
    padded = [BOS] + x[:max_len] + [EOS] + [PAD] * max(0, max_len - len(x))
    length = 1+min(len(x), max_len)+1
    return np.array(padded), length

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>')
    batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def __len__(self):
    # if self.is_train:
    #   return len(self.captions)
    # else:
    #   return len(self.names)
    return len(self.ref_captions)

  def __getitem__(self, idx):
    outs = {}
    # if self.is_train:
    #   name = self.names[self.cap2ftid[idx]]
    # else:
    #   name = self.names[idx]
    # name = list(self.ref_captions.keys())[idx]
    name = self.names[idx]
    example = self.ref_captions[name]
    # if self.is_train:
    #   example = self.train_ref[idx]
    # else:
    #   example = self.val_tst_ref[idx]

    # example = self.ref_captions[idx]
    # start = int(example["timestamps"][0][0])
    # end = int(example["timestamps"][0][1])
    sentence = example["sentences"][0]
    movie_id = example["movie_id"]
    movie_idx = self.movie2id[movie_id]
    feat_path_resnet = os.path.join(self.ft_root, "resnet_clip/{}.npy.npz".format(name))
    feat_path_s3d = os.path.join(self.ft_root, "s3d_clip/{}.npy.npz".format(name))
    # print(feat_path_resnet)
    raw_feat_resnet = np.load(feat_path_resnet)['features']
    raw_feat_s3d = np.load(feat_path_s3d)['features']
    raw_feat = np.concatenate((raw_feat_resnet, raw_feat_s3d), axis=-1)

    feat_len = raw_feat.shape[0]
    feat_dim = raw_feat.shape[1]

    max_v_l = 25

    video_feature = np.zeros((max_v_l, feat_dim), np.float32)  # only video features and padding
    video_feature[:feat_len] = raw_feat[:]

    outs['movie_id'] = movie_idx
    outs['ft_len'] = feat_len
    outs['img_ft'] = video_feature
    outs['name'] = name

    if self.is_train:
      outs['ref_sents'] = sentence
      sent_id, sent_len = self.pad_sent(self.sent2int(sentence))
      outs['caption_ids'] = sent_id
      outs['id_len'] = sent_len
    return outs
