## Emotion
修士論文のシミュレーションコードです。</p>
以下、各フォルダおよびファイルの説明になります。

### observable_ER
感情表現eと感受性rの両方が観察できるモデル。</p>
戦略決定に関しては、ナッシュ均衡を直接計算。

### observable_E
感情表現eのみが観察できるモデル。</p>
戦略決定に関しては、学習モデルを使用。

### observable_R
感受性rのみが観察できるモデル。</p>
戦略決定に関しては、学習モデルを使用。

### Unobservable
感情表現eと感受性rのいずれも観察できないモデル。</p>
戦略決定に関しては、学習モデルを使用。

### analysis.py
出力ファイルの分析用コード

