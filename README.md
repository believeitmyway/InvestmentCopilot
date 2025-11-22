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
- `GOOGLE_API_KEY` / `GOOGLE_GENAI_API_KEY` / `GENAI_API_KEY` / `GEMINI_API_KEY` *(任意)*: いずれかが設定されていれば Gemini 用の入力欄に自動で反映されます。名称の異なる既存プロジェクトからでもそのまま流用できます。

## Gemini キーとモデル指定
1. Google AI Studio で API キーを発行し、上記いずれかの環境変数に保存するか、アプリ起動後に「Google AI Studio API Key」欄へ貼り付けます。
2. 「Gemini モデルID」欄で使用したいモデル名を指定してください。デフォルトは `gemini-1.5-flash` ですが、`gemini-1.5-pro` など任意のサポートモデルへ切り替え可能です。
3. API キーが有効であれば Gemini レスポンスが最優先で使用され、失敗した場合のみ OpenAI → ヒューリスティックの順でフォールバックします。

## Chrome パスワードマネージャー連携
- OpenAI / Google AI Studio の API キー、Gemini モデルID 各欄には `name` / `autocomplete` 属性を設定しているため、Google Chrome のパスワードマネージャーに資格情報として保存・自動入力できます。
- Chrome が表示する保存ポップアップで許可すれば、次回からワンクリックで復元できます。
- キーやモデルIDはブラウザ内でのみ使用され、サーバー側へは保存されません。

## API キー適用の流れ
1. 各 API キーとモデル ID を入力します。
2. 「APIキーを適用して画面下部を更新」ボタンを押して、分析に使用する値を確定します。
3. 画面下部の「API設定ステータス」カードに、設定済み / 未設定 と最終適用時刻が表示されます。

## 注意事項
- 本アプリは教育目的の情報提供ツールです。最終的な投資判断はご自身の責任で行ってください。