import numpy as np
import librosa as lib

def main():
    path = r"sound_file\ヨルシカ - 藍二乗 (Music Video) [4MoRLTAJY_0].wav"
    data, fs = lib.load(path)

    # ピッチ
    # ピッチ=基本周波数(F0) stftの結果各フレームごとにピッチを算出 -> numpy配列に代入(各列がフレーム、各行が周波数に対応)
    pitches, magnitudes = lib.core.piptrack(y=data, sr=fs)
    for t in range(pitches.shape[1]):
        
        



if __name__ == "__main__":
    main()