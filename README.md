1. NewProject/datas/* ディレクトリのファイルを展開します。
2. train.pyを実行してモデルの学習を開始する(*)
3.結果を表示し、以下のファイルが正常に保存されていることを確認します。
   - history3d.json: 学習時の損失データ
   - model3d.h5: ニューラルネットワークパラメータ
   - dice.png：1フレームあたりの回復精度（DSC）のタイムラプス画像
4.test.pyのテスト実行

(*): 実行前に anaconda で tensorflow-gup 環境を設定する必要があります。
