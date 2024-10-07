import librosa as lib
import numpy as np
import os
import matplotlib.pyplot as plt

# 音声ファイル読み込んでメルスペクトログラム化[単一の音声ファイルパス -> dbスケールメルスペクトログラム, サンプリング周波数]
def create_spectrogram(path:str):
    # 音声ファイル読み込み
    y, sr = lib.load(path, sr=None)

    # メルスペクトログラム作成
    spectrogram = lib.feature.melspectrogram(y=y, sr=sr)

    # dbスケール変換
    spec_db = lib.power_to_db(spectrogram, ref=np.max)

    return spec_db, sr

# スペクトログラム表示[スペクトログラム(np配列), サンプリング周波数]
def show_spectrogram(spec_db:np.ndarray, sr:int):
    


# 変数の詳細を出力(テスト用)[何らかの変数]
def test_output(output_variable):
    # 変数詳細出力
    print(f"data:{output_variable}")
    print(f"Shape:{output_variable.shape}")

    return 

# テスト関数(好きにいじる)
def test():
    spec_db, sr = create_spectrogram(r"C:\Users\重岡拓郎\Desktop\code\voice\sound_file\iyowa\1000年生きてる.wav")
    test_output(spec_db)


# メイン関数
def main():
    print("test")

# 開始部
if __name__ == "__main__":
    #test()
    main()