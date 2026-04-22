# ⚙ GearMesh Viewer

平歯車のバックラッシ計算とかみ合い可視化を行うWebアプリです。

## 機能

- モジュール、圧力角、歯数、転位係数からギアかみ合いを描画
- 跨ぎ歯厚 (W) 実測値からバックラッシ (jt, bn) を算出
- インボリュート歯形の干渉検出・可視化
- G2追加回転による干渉シミュレーション
- かみ合い率 εα の算出

## 使い方

### オンライン（インストール不要）

Streamlit Community Cloud でホストされています：
<!-- デプロイ後にURLを記入 -->

### ローカル実行

```bash
pip install -r requirements.txt
streamlit run gerar_span_web.py
```

ブラウザで http://localhost:8501 が開きます。

## 入力パラメータ

| パラメータ | 説明 |
|---|---|
| m | モジュール [mm] |
| α | 圧力角 [deg] |
| z1, z2 | 歯数 |
| x1, x2 | 転位係数（設計値） |
| k1, k2 | 跨ぎ歯数 |
| W1, W2 | 跨ぎ歯厚 [mm] |
| a | 軸間距離 [mm] |
| G2追加回転 | G2の追加回転角度 [deg] |

## 必要環境

- Python 3.10+
- streamlit, matplotlib, numpy
