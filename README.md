# Mobile-First AI Investment Dashboard

忙しいビジネスマンがスマホの1画面で「今すぐ売買すべきか」を判断できるよう設計した Streamlit アプリです。`yfinance` で取得した株価・アナリスト評価と、`duckduckgo-search` による最新ニュースを集約し、OpenAI LLM がワンフレーズ結論と 3 つの要点を提示します。

## 主な機能
- **モバイル最適**: シングルカラム × スティッキーヘッダーで、縦スクロールのみで重要情報を確認。
- **AI投資スコア**: OpenAI (gpt-4o-mini) が結論・スコア・Bull/Bearシナリオを JSON で返答。
- **アナリスト比較**: `recommendationKey`, `targetMeanPrice`, `numberOfAnalystOpinions` を可視化し、AIコメントと並列表示。
- **ニュース統合**: DuckDuckGo から関連ヘッドラインを取得し、タップで外部記事へ遷移。
- **フォールバック分析**: APIキー未入力時は指標ベースのヒューリスティック推奨を自動生成。

## セットアップ
```bash
pip install -r requirements.txt
```

## 実行方法
```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開き、ティッカー（例: `AAPL`）と任意で OpenAI API Key を入力してください。APIキーはブラウザ内でのみ使用され、サーバーには保存されません。

## 環境変数
- `OPENAI_API_KEY` *(任意)*: 設定済みの場合、アプリの入力欄に自動で反映されます。

## 注意事項
- 本アプリは教育目的の情報提供ツールです。最終的な投資判断はご自身の責任で行ってください。