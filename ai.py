import librosa as lib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim


#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

# 各アーティストのクラス (これに対してai処理を行うべき？) ()
class Artist:
    # イニシャライザ [アーティスト名]
    def __init__(self, name):
        self.name = name      # アーティスト名
        self.audio_files = [] # AudioFileインスタンスをadd_audiofileで格納していく

    # AudioFileインスタンスをaudio_filesに追加 (基本的にall_addから呼び出すからメソッドにする必要ない...?)[AudioFileインスタンス]
    def add_audiofile(self, audiofile):
        # リスト追加(np配列ではない)
        self.audio_files.append(audiofile)

    # pathディレクトリ内のwavをすべてAudioFile化して、それぞれadd_audiofileに与える。要は今のアーティストのArtistを完成させる [アーティストディレクトリまでのパス, アーティストディクレトリパス]
    def all_add(self, base, path):
        # パス結合(アーティストディレクトリまでのパス)
        full_path  = os.path.join(base, path)

        # ディクレトリ内のファイル(とディレクトリ)すべてでループ
        for track in os.listdir(full_path):
            # .wavファイルかどうかチェックするよ～
            if track.endswith(".wav"):
                # 曲名生成(trackが"曲名.wav"の形式なので、".wav"を""に置き換える)
                track_name = track.replace(".wav", "")

                # 曲のパス生成(現在のtrackまでのパス)
                song_path = os.path.join(full_path, track)

                # 今のtrackのAudioFileインスタンス作成
                audio = AudioFile(path, track_name, song_path)

                # AudioFileをself.audio_filesに追加していく
                self.add_audiofile(audio)

    # AudioFile数出力
    def print_len(self):
        print(len(self.audio_files))

    # transform_into_standard_spectrogramを全audiofileに実行
    def all_transform(self):
        for audio in self.audio_files:
            audio.transform_into_standard_spectrogram()
            audio.transform_into_tensor()

#---------------------------------------------------------------------------------#

# 各音声ファイルのクラス
class AudioFile:
    # イニシャライザ [アーティスト名、曲名、パス] 曲名に関してはパスから引き出せるかも
    def __init__(self, artist_name, track_name, file_path):
        self.artist_name = artist_name # アーティスト名
        self.track_name = track_name   # 曲名
        self.file_path = file_path     # wavファイルパス
        self.sr = None                 # サンプリング周波数
        self.spectrogram = None        # dbスケールのメルスペクトログラム
        self.tensor = None             # cnn用のspectrogramをtensor型に変換したもの 最初はNoneのまま
        
        # 音声ファイル読み込みとスペクトログラム[sr, spectrogram設定]
        self.create_spectrogram()

    # 音声ファイル読み込んでメルスペクトログラム化[単一の音声ファイルパス -> dbスケールメルスペクトログラム, サンプリング周波数]
    def create_spectrogram(self):
        # 音声ファイル読み込み
        y, sr = lib.load(self.file_path, sr=None)

        # サンプリング周波数設定
        self.sr = sr

        # メルスペクトログラム作成
        spectrogram = lib.feature.melspectrogram(y=y, sr=sr)

        # dbスケール変換
        spec_db = lib.power_to_db(spectrogram, ref=np.max)

        # スペクトログラム設定
        self.spectrogram = spec_db

    # スペクトログラム表示[スペクトログラム(np配列), サンプリング周波数]
    def show_spectrogram(self):
        # グラフ作成
        plt.figure()

        # スペクトログラム描画
        lib.display.specshow(self.spectrogram, sr=self.sr, x_axis="time", y_axis="mel")

        # 詳細追加
        plt.title("Mel-frequency spectrogram")
        plt.tight_layout()

        # 表示
        plt.show()

    # スペクトログラム正規化(基本的にArtist.all_transformから呼び出す)
    def transform_into_standard_spectrogram(self):
        scaler = StandardScaler()
        self.spectrogram = scaler.fit_transform(self.spectrogram)

    # cnn用にスペクトログラムをtensorに変換
    def transform_into_tensor(self):
        trans = torchvision.transforms.ToTensor()
        self.tensor = trans(np.array(self.spectrogram))

#---------------------------------------------------------------------------------#

# 自作データセット型(torch.utils.data.Datasetを継承) [Artistのリスト]
class MusicDataset(Dataset):
    def __init__(self, artist_list):
        self.data = []
        self.labels = []
        self.dataloader = None


        # 曲とラベルを集約(ここでのラベルはアーティスト名)
        for artist in artist_list:
            for audio in artist.audio_files:
                self.data.append(audio.tensor)
                self.labels.append(artist.name)
        
    # データローダー作成(DataLoader:「全データをまとめたリスト と それに対応するラベルのリスト」をもつクラス)
    def make_dataloader(self, batch_size):
        self.dataloader = DataLoader(self, batch_size=batch_size, shuffle=False)

    # 学習
    def model_train(self, epoch_num, optimizer):
        losses = []

        for epoch in range(epoch_num):

            # 学習
            train_losses = 0

            for data in self.dataloader:
                # データごとに勾配初期化を明示する必要があるっぽい？
                optimizer.zero_grad()

                # 恐らく x=データ本体 y=ラベル
                x, y = data

                # 各種データをGPUへ
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)

                print(x)
                print(y)
                exit()




                

            
#---------------------------------------------------------------------------------#

# 変数の詳細を出力(テスト用)[何らかの変数]
def test_output(output_variable):
    # 変数詳細出力
    print(f"data:{output_variable}")
    print(f"Shape:{output_variable.shape}")

    return 

# テスト関数(好きにいじる)
def test():

    iyowa = Artist("iyowa")
    inabakumori = Artist("inabakumori")
    aoya = Artist("aoya")
    print("1")
    iyowa.all_add("sound_file", "iyowa")
    inabakumori.all_add("sound_file", "inabakumori")
    aoya.all_add("sound_file", "aoya")
    print("2")
    iyowa.all_transform()
    inabakumori.all_transform()
    aoya.all_transform()
    print("3")
    train = MusicDataset([iyowa, inabakumori, aoya])
    print("4")
    train.make_dataloader(32)
    print("5")

    # 学習済みモデルをさらに学習(新規データで学習していく 学習済モデルをモデルとして学習していく)
    resnet_model = resnet34(pretrained=True)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model.fc = nn.Linear(512, 3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)

    # 学習率
    lr = 2e-4

    # 損失関数と最適化手法 (「出力と教師信号の誤差」を損失関数で求め、それをbpして、最適化手法により重み(cnnの場合はフィルタ)更新 今回のadamは学習率も更新してる？)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=lr)


    # バッチサイズ:学習前にデータセットをいくつかのサブセットに分ける。それぞれのサブセットに含まれるデータ数のこと。2^nが望ましい
    # イテレーション数:データセット全体が学習されるのに必要な学習回数のこと。要は データセットサイズをバッチサイズで割ったもの(の切り上げ？)
    # エポック数:全体の学習を何回やるか。「N個のサブセットに分けN回学習をする」を何回やるか。
    epochs = 50

    #train.lr_decay()

    # 学習(エラー中)
    train.model_train(epoch_num=epochs, optimizer=optimizer)
    
#---------------------------------------------------------------------------------#

# メイン関数
def main():
    print("test")

# 開始部
if __name__ == "__main__":
    test()
    main()