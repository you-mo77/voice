import numpy as np
import PySimpleGUI as sg

# gui生成
def gui():
    # guiレイアウト作成(各軸の横にプルダウンで主成分軸とクラスタ数選択させる)
    layout = [[sg.Text("横軸"), sg.Text("縦軸")],
              [sg.Image(filename="", key="-image-")],
              [sg.Button("Run")]]
    
    # ウィンドウ作成
    window = sg.Window("主成分プロット", layout)

    # イベントループ
    while True:
        # 読み込み
        event, values = window.read()

        # 終了
        if event == sg.WIN_CLOSED:
            break


# メイン
def main():
    # 一時
    print()

    # gui表示
    gui()

# 実行部分
if __name__ == "__main__":
    main()