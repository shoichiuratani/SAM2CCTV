# デプロイメントガイド

このドキュメントでは、Grounded-SAM-2動画分析アプリケーションをGoogle Cloud Platformにデプロイする手順を説明します。

## 📋 前提条件

### 必要なツール
- Google Cloud SDK (gcloud)
- Docker
- kubectl
- Git

### Google Cloudアカウント
- 有効なGoogle Cloudアカウント
- 課金が有効なGCPプロジェクト
- 必要な権限（Compute Admin、Kubernetes Engine Admin、Cloud Run Admin等）

## 🚀 クイックスタート

### 1. 自動デプロイ（推奨）

```bash
# 1. プロジェクトの設定
gcloud config set project YOUR_PROJECT_ID

# 2. GCPセットアップの実行
cd deploy
./gcp-setup.sh

# 3. 全体デプロイの実行
./deploy-all.sh
```

### 2. ローカル開発環境のセットアップ

```bash
# ローカル開発環境を構築
cd deploy
./local-dev.sh

# Supervisorでサービス起動
cd ..
supervisord -c supervisord.conf
supervisorctl -c supervisord.conf status

# Streamlitアプリの起動
cd streamlit_app
INFERENCE_SERVER_URL=http://localhost:8080/predict COBOL_API_URL=http://localhost:8081/get_product_info streamlit run streamlit_app.py
```

## 📖 詳細なデプロイ手順

### Phase 1: Google Cloud環境の準備

#### 1.1 プロジェクトの設定
```bash
# プロジェクトIDを設定
gcloud config set project YOUR_PROJECT_ID

# 現在の設定を確認
gcloud config list
```

#### 1.2 必要なAPIの有効化
```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com
```

#### 1.3 Artifact Registryリポジトリの作成
```bash
gcloud artifacts repositories create grounded-sam2-repo \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="Grounded-SAM-2 Application Repository"

# Docker認証の設定
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

### Phase 2: AI推論サーバーのデプロイ

#### 2.1 Dockerイメージのビルド・プッシュ
```bash
cd inference_server

# Cloud Buildを使用したビルド・プッシュ
gcloud builds submit --tag asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/inference-server:v1 .
```

#### 2.2 GKEクラスタの作成
```bash
# Autopilotモードでクラスタを作成
gcloud container clusters create-auto ai-cluster --region asia-northeast1

# kubectl認証情報を取得
gcloud container clusters get-credentials ai-cluster --region asia-northeast1
```

#### 2.3 Kubernetesリソースのデプロイ
```bash
# デプロイメントファイルのプロジェクトID更新
sed "s/PROJECT_ID/YOUR_PROJECT_ID/g" k8s/deployment.yaml > k8s/deployment-updated.yaml

# リソースをデプロイ
kubectl apply -f k8s/deployment-updated.yaml

# デプロイメント状況の確認
kubectl get pods
kubectl get services

# 外部IPの取得（数分かかる場合があります）
kubectl get service inference-server-service --watch
```

### Phase 3: Webアプリケーションのデプロイ

#### 3.1 Dockerイメージのビルド・プッシュ
```bash
cd ../streamlit_app

# Cloud Buildを使用したビルド・プッシュ
gcloud builds submit --tag asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/streamlit-app:v1 .
```

#### 3.2 Cloud Runへのデプロイ
```bash
# AI推論サーバーの外部IPを取得
INFERENCE_SERVER_IP=$(kubectl get service inference-server-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Cloud Runサービスをデプロイ
gcloud run deploy streamlit-app \
  --image asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/streamlit-app:v1 \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "INFERENCE_SERVER_URL=http://${INFERENCE_SERVER_IP}/predict"
```

## 🔧 運用管理

### サービス状態の確認

#### GKE（AI推論サーバー）
```bash
# Pod状態の確認
kubectl get pods -l app=inference-server

# サービス状態の確認
kubectl get services

# ログの確認
kubectl logs -l app=inference-server

# リソース使用状況の確認
kubectl top pods
```

#### Cloud Run（Webアプリケーション）
```bash
# サービス一覧の確認
gcloud run services list --region=asia-northeast1

# サービス詳細の確認
gcloud run services describe streamlit-app --region=asia-northeast1

# ログの確認
gcloud run services logs read streamlit-app --region=asia-northeast1
```

### スケーリング

#### GKE
```bash
# レプリカ数の変更
kubectl scale deployment inference-server --replicas=3

# HPA（水平オートスケーリング）の設定
kubectl autoscale deployment inference-server --cpu-percent=70 --min=2 --max=10
```

#### Cloud Run
```bash
# 同時実行数とインスタンス数の調整
gcloud run services update streamlit-app \
  --region=asia-northeast1 \
  --concurrency=100 \
  --min-instances=1 \
  --max-instances=10
```

### 更新とロールバック

#### アプリケーションの更新
```bash
# 新しいイメージをビルド・プッシュ
gcloud builds submit --tag asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/inference-server:v2 ./inference_server

# GKEのデプロイメントを更新
kubectl set image deployment/inference-server inference-server=asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/inference-server:v2

# Cloud Runサービスを更新
gcloud run deploy streamlit-app \
  --image asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/grounded-sam2-repo/streamlit-app:v2 \
  --region=asia-northeast1
```

#### ロールバック
```bash
# GKEのロールバック
kubectl rollout undo deployment/inference-server

# Cloud Runのロールバック（リビジョン指定）
gcloud run services update-traffic streamlit-app \
  --to-revisions=REVISION_NAME=100 \
  --region=asia-northeast1
```

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. 外部IPが割り当てられない
```bash
# サービスタイプの確認
kubectl get service inference-server-service -o yaml

# Load Balancerの状態確認
kubectl describe service inference-server-service

# 必要に応じてサービスを削除・再作成
kubectl delete service inference-server-service
kubectl expose deployment inference-server --type=LoadBalancer --port 80 --target-port 8080
```

#### 2. Pod が起動しない
```bash
# Pod の詳細確認
kubectl describe pod POD_NAME

# Pod のログ確認
kubectl logs POD_NAME

# イベントの確認
kubectl get events --sort-by=.metadata.creationTimestamp
```

#### 3. Cloud Run サービスでタイムアウト
```bash
# タイムアウト設定の確認・変更
gcloud run services update streamlit-app \
  --region=asia-northeast1 \
  --timeout=600

# メモリとCPUの増強
gcloud run services update streamlit-app \
  --region=asia-northeast1 \
  --memory=4Gi \
  --cpu=4
```

#### 4. イメージプルエラー
```bash
# Docker認証の再設定
gcloud auth configure-docker asia-northeast1-docker.pkg.dev

# サービスアカウントの権限確認
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### ログの確認方法

#### 集約ログの確認
```bash
# Cloud Loggingでフィルタリング
gcloud logging read "resource.type=k8s_container AND resource.labels.container_name=inference-server" --limit=50

gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=streamlit-app" --limit=50
```

## 🛡️ セキュリティ対策

### 本番環境での推奨事項

#### 1. IAMとサービスアカウント
```bash
# 専用サービスアカウントの作成
gcloud iam service-accounts create grounded-sam2-sa \
  --display-name="Grounded-SAM-2 Service Account"

# 最小権限の付与
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:grounded-sam2-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

#### 2. ネットワークセキュリティ
```bash
# プライベートクラスタの作成
gcloud container clusters create ai-cluster-private \
  --enable-private-nodes \
  --master-ipv4-cidr 172.16.0.0/28 \
  --region asia-northeast1

# ファイアウォールルールの設定
gcloud compute firewall-rules create allow-inference-server \
  --allow tcp:8080 \
  --source-ranges 10.0.0.0/8 \
  --target-tags inference-server
```

#### 3. 認証の追加
```bash
# Cloud Run での認証有効化
gcloud run services update streamlit-app \
  --region=asia-northeast1 \
  --no-allow-unauthenticated

# IAP（Identity-Aware Proxy）の設定
gcloud compute backend-services update BACKEND_SERVICE \
  --iap=enabled
```

## 💰 コスト最適化

### 推奨設定

#### 1. リソース制限
```yaml
# Kubernetes リソース制限の例
resources:
  requests:
    memory: "500Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### 2. オートスケーリング
```bash
# GKE Autopilot使用（リソース効率化）
gcloud container clusters create-auto ai-cluster \
  --region asia-northeast1

# Cloud Run の最小インスタンス数を0に設定
gcloud run services update streamlit-app \
  --region=asia-northeast1 \
  --min-instances=0
```

#### 3. スケジューラーベースの停止
```bash
# Cloud Schedulerで夜間停止（開発環境）
gcloud scheduler jobs create http scale-down-job \
  --schedule="0 22 * * *" \
  --uri="https://container.googleapis.com/v1/projects/YOUR_PROJECT_ID/zones/asia-northeast1/clusters/ai-cluster" \
  --http-method=PUT
```

## 📚 参考資料

- [Google Kubernetes Engine ドキュメント](https://cloud.google.com/kubernetes-engine/docs)
- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs)
- [Cloud Build ドキュメント](https://cloud.google.com/build/docs)

## 🆘 サポート

技術的な問題やデプロイに関する質問がある場合は、プロジェクトチームまでお問い合わせください。

---

最終更新日: 2024-09-22