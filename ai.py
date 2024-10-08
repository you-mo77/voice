import librosa as lib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#---------------------------------------------------------------------------------#

# 各アーティストのクラス (これに対してai処理を行うべき？)
class Artist:
    # イニシャライザ [アーティスト名]
    def __init__(self, name):
        self.name = name      # アーティスト名
        self.audio_files = [] # AudioFileインスタンスを随時格納していく

    # AudioFileインスタンスをaudio_filesに追加 [AudioFileインスタンス]
    def add_audiofile(self, audiofile):
        # リスト追加(np配列ではない)
        self.audio_files.append(audiofile)

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

#---------------------------------------------------------------------------------#

# 変数の詳細を出力(テスト用)[何らかの変数]
def test_output(output_variable):
    # 変数詳細出力
    print(f"data:{output_variable}")
    print(f"Shape:{output_variable.shape}")

    return 

# テスト関数(好きにいじる)
def test():
    test_audio = AudioFile("iyowa", "1000年生きてる", r"sound_file\iyowa\1000年生きてる.wav")
    test_audio.show_spectrogram()
    

# メイン関数
def main():
    print("test")

# 開始部
if __name__ == "__main__":
    test()
    main()