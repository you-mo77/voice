import librosa as lib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    iyowa = Artist("iyowa")
    aoya = Artist("aoya")
    inabakumori = Artist("inabakumori")

    iyowa.all_add("sound_file", "iyowa")
    aoya.all_add("sound_file", "aoya")
    inabakumori.all_add("sound_file", "inabakumori")

    #a = os.path.join("sound_file", "iyowa")
    #pa = os.path.join(pa, "たぶん終わり.wav")

    #print(pa)
    #t = AudioFile("iyowa", "たぶん終わり", pa)
    #print(t.track_name)
    
    print(iyowa.print_len())
    print(aoya.print_len())
    print(inabakumori.print_len())
    print(iyowa.audio_files)
    

# メイン関数
def main():
    print("test")

# 開始部
if __name__ == "__main__":
    test()
    main()