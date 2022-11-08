import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import librosa
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from IPython.display import Audio


class MyDataSet(torch.utils.data.Dataset):

  def __init__(self, data_json,resample_rate,rawMode=False):
        self.resample_rate=resample_rate
        self.data = pd.read_json(data_json, lines=False)
        self.data=pd.json_normalize(self.data.data)
        label_Encoder=LabelEncoder()
        encode_labels=label_Encoder.fit_transform(self.data.voice)
        self.data["label"] = encode_labels
        self.rawMode=rawMode

  def __len__(self):
        return len(self.data)


  def __getitem__(self, idx):
      def MFCC(waveform,sample_rate):
        n_fft = 2048
        win_length = None
        hop_length = 256
        n_mels = 128
        n_mfcc = 128
        melspec=librosa.feature.melspectrogram(y=waveform.numpy()[0],
                                               sr=sample_rate,
                                               n_fft=n_fft,win_length=win_length,
                                               hop_length=hop_length,
                                               n_mels=n_mels,htk=True,
                                               norm=None,)
        return  librosa.feature.mfcc(S=librosa.core.spectrum.power_to_db(melspec),
                                     n_mfcc=n_mfcc,
                                     dct_type=2,
                                     norm="ortho",)
      if torch.is_tensor(idx):
            idx = idx.item()

      try:    

            file_path = self.data.path.iloc[idx]
            waveform, sample_rate = torchaudio.load(os.path.join("./soundWake/",str(file_path)))
            if sample_rate != self.resample_rate:
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate,new_freq=self.resample_rate)
                waveform = transform(waveform)
            waveform = torch.mean(waveform,
                            dim=0, keepdim=True)
            label = torch.tensor(self.data.label.iloc[idx])
            if(not self.rawMode):
              fixed_length = 128
              waveform=torchaudio.transforms.MFCC(self.resample_rate,n_mfcc=128,log_mels=True)(waveform)
              if waveform.shape[2] < fixed_length:
                waveform = torch.nn.functional.pad(
                waveform, (0, fixed_length -waveform.shape[2]))
                # print("less than")
              else:
                waveform =waveform[:,:, :fixed_length]
                # print("more than")

      except Exception as e:
            print(str(e))
            return e
      return waveform,label


