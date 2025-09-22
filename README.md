# Grounded-SAM-2 動画分析アプリケーション

## プロジェクト概要

企業の基幹システム（COBOL製API）と最新のAIモデル「Grounded-SAM-2」を連携させ、アップロードされた動画を分析するWebアプリケーションです。Google Cloud Platform上にデプロイして運用します。

## アーキテクチャ

本プロジェクトは、以下の3つの独立したコンポーネントで構成されます。

### 1. AI推論サーバー
- **場所**: `inference_server/`
- **役割**: Grounded-SAM-2モデルを実行し、外部からのリクエストに応じて動画分析を行うAPIサーバー
- **デプロイ先**: Google Kubernetes Engine (GKE)
- **技術スタック**: Python, Flask, Grounded-SAM-2, OpenCV

### 2. Webアプリケーション
- **場所**: `streamlit_app/`
- **役割**: ユーザーが操作するUI。動画のアップロードと結果表示を行う
- **デプロイ先**: Cloud Run
- **技術スタック**: Python, Streamlit, OpenCV

### 3. COBOL連携API（既存）
- **役割**: 製品情報などを返すレガシーシステムのAPI
- **実装**: 本プロジェクトではこのAPIを呼び出す側を実装

## 機能

- 動画ファイルのアップロード（MP4, MOV, AVI対応）
- COBOL APIとの連携による製品情報の取得
- Grounded-SAM-2を使用したオブジェクト検出
- 検出結果を重畳した動画の生成・表示
- リアルタイムでの分析結果表示

## 技術スタック

### バックエンド
- Python 3.10
- Flask (AI推論サーバー)
- Streamlit (Webアプリケーション)
- Grounded-SAM-2
- OpenCV
- PyTorch

### インフラストラクチャ
- Google Cloud Platform (GCP)
- Google Kubernetes Engine (GKE)
- Cloud Run
- Artifact Registry
- Cloud Build

### コンテナ化
- Docker
- Kubernetes

## ディレクトリ構造

```
.
├── README.md
├── inference_server/
│   ├── inference_server.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── k8s/
│       └── deployment.yaml
├── streamlit_app/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
└── deploy/
    ├── gcp-setup.sh
    └── deploy-all.sh
```

## セットアップ手順

### 1. 前提条件
- Google Cloud SDKのインストールと認証
- Dockerのインストール
- kubectlのインストール（GKE用）

### 2. Google Cloudプロジェクトの設定
```bash
# プロジェクトIDを設定
gcloud config set project YOUR_PROJECT_ID

# 必要なAPIを有効化
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  run.googleapis.com
```

### 3. AI推論サーバーのデプロイ
```bash
# コンテナイメージをビルド・プッシュ
gcloud builds submit ./inference_server --tag gcr.io/$(gcloud config get-value project)/inference-server:v1

# GKEクラスタを作成
gcloud container clusters create-auto ai-cluster --region asia-northeast1

# GKEにデプロイ
kubectl create deployment inference-server --image=gcr.io/$(gcloud config get-value project)/inference-server:v1
kubectl expose deployment inference-server --type=LoadBalancer --port 80 --target-port 8080
```

### 4. Webアプリケーションのデプロイ
```bash
# コンテナイメージをビルド・プッシュ
gcloud builds submit ./streamlit_app --tag gcr.io/$(gcloud config get-value project)/streamlit-app:v1

# Cloud Runにデプロイ
gcloud run deploy streamlit-app \
  --image gcr.io/$(gcloud config get-value project)/streamlit-app:v1 \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --set-env-vars="INFERENCE_SERVER_URL=http://YOUR_AI_SERVER_EXTERNAL_IP/predict"
```

## 使用方法

1. WebアプリケーションのURLにアクセス
2. 製品ID（任意）を入力してCOBOL APIから製品情報を取得
3. 動画ファイルをアップロード
4. 検出したいオブジェクト名を入力
5. 「分析を実行」ボタンをクリック
6. AI推論結果が重畳された動画を確認

## 開発者向け情報

### ローカル開発

#### AI推論サーバー
```bash
cd inference_server
pip install -r requirements.txt
python inference_server.py
```

#### Webアプリケーション
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Grounded-SAM-2の実装について

現在の実装はシミュレーション版です。本格運用では以下の手順でGrounded-SAM-2を統合してください：

1. 公式リポジトリからGrounded-SAM-2をクローン
2. モデルの重みファイルをダウンロード
3. `inference_server.py`の推論部分を実際のモデル呼び出しに置換
4. 必要な依存関係を`requirements.txt`に追加

## 注意事項

- AI推論サーバーはGPUインスタンスでの実行を推奨
- 動画ファイルサイズに応じてメモリ使用量を調整
- COBOL APIの仕様に応じてリクエスト形式を調整
- 本番環境では適切な認証・認可機能の実装が必要

## ライセンス

このプロジェクトは内部使用を目的として開発されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。開発に参加される際は、以下のガイドラインに従ってください：

1. genspark_ai_developerブランチで作業
2. 機能ごとに適切なコミットメッセージを作成
3. テスト実行後にプルリクエストを作成

## サポート

技術的な質問や問題については、プロジェクトチームまでお問い合わせください。