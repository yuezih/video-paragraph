from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import math
import torch.utils.data
import h5py

UNK, PAD, BOS, EOS = 0, 1, 2, 3


class CaptionDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_root, cap_file, word2int, int2word,
    max_words_in_sent=150, is_train=False, _logger=None):
    super(CaptionDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info


    self.ref_captions = json.load(open(cap_file)) # gt
    self.names = list(self.ref_captions.keys())
    self.meta_anno = json.load(open('/data2/yzh/Dataset/MOVIES/metadata/meta_anno.json'))
    self.face_feature_h5 = h5py.File('/data2/yzh/Dataset/MOVIES/metadata/face_profile_512dim.hdf5', 'r')
    self.movie_with_face_list = os.listdir('/data2/yzh/Dataset/MOVIES/metadata/xigua_face')
    self.stoi = json.load(open(word2int))
    self.itos = json.load(open(int2word))
    self.ft_root = ft_root
    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    # self.rolestoi = json.load(open('/data2/yzh/Dataset/MOVIES/annotation/811/vocab/rolename/rolename_vocab/c2id.json'))


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
    name = self.names[idx]
    example = self.ref_captions[name]
    sentence = example["sentences"][0]

    roles_feature = np.zeros((3, 512), np.float32)
    rolenames_id_sent = []
    role_list = []

    movie_id = example["movie_id"]
    if movie_id in self.meta_anno and movie_id in self.movie_with_face_list:
      role_list = [role_id for role_id in self.meta_anno[movie_id]][:3]

      for i in range(len(role_list)):
        role_id = role_list[i]
        # rolename
        rolename = self.meta_anno[movie_id][role_id]["rolename"]
        rolename_split = i+4 # <role{i+1}>
        rolenames_id_sent.append(rolename_split)
        rolenames_id_sent += [self.stoi.get(w, UNK) for w in rolename]
        # faces feature
        if role_id in self.face_feature_h5:
          role_feature = np.array(self.face_feature_h5[role_id]['features'], dtype=np.float32)
          roles_feature[i] = role_feature
    
    rolename_len = len(rolenames_id_sent)
    rolenames_id_sent += [1] * (20 - rolename_len)
    rolenames_id_sent = np.array(rolenames_id_sent[:20])

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

    outs['ft_len'] = feat_len
    outs['img_ft'] = video_feature
    outs['name'] = name
    outs['face_ft'] = roles_feature
    outs['face_ft_len'] = len(role_list)
    outs['rolename_id'] = rolenames_id_sent
    outs['rolename_len'] = rolename_len

    if self.is_train:
      outs['ref_sents'] = sentence
      sent_id, sent_len = self.pad_sent(self.sent2int(sentence))
      outs['caption_ids'] = sent_id
      outs['id_len'] = sent_len
    return outs
