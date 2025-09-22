#!/bin/bash

# Grounded-SAM-2 動画分析アプリケーション デプロイスクリプト
# このスクリプトは、AI推論サーバーとWebアプリケーションをGoogle Cloudにデプロイします。

set -e

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ出力関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 環境変数の読み込み
if [ -f "../.env" ]; then
    source "../.env"
    log_info "環境変数を読み込みました"
else
    log_error ".envファイルが見つかりません。最初にgcp-setup.shを実行してください。"
    exit 1
fi

# プロジェクト設定の確認
CURRENT_PROJECT=$(gcloud config get-value project)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    log_error "GCPプロジェクトが一致しません。現在: $CURRENT_PROJECT, 期待値: $PROJECT_ID"
    exit 1
fi

log_info "デプロイ開始 - プロジェクト: $PROJECT_ID"
echo ""

# ===== Phase 1: AI推論サーバーのデプロイ =====
log_info "=== Phase 1: AI推論サーバーのデプロイ ==="

# Docker イメージのビルド・プッシュ
log_info "AI推論サーバーのDockerイメージをビルド中..."
cd ../inference_server
gcloud builds submit --tag "$INFERENCE_SERVER_IMAGE" .
log_success "AI推論サーバーのイメージをArtifact Registryにプッシュしました"

# GKEクラスタの作成
log_info "GKEクラスタを作成中..."
if gcloud container clusters describe "$GKE_CLUSTER_NAME" --zone="$GKE_ZONE" >/dev/null 2>&1; then
    log_warning "GKEクラスタ '$GKE_CLUSTER_NAME' は既に存在します"
else
    gcloud container clusters create-auto "$GKE_CLUSTER_NAME" --region="$GKE_ZONE"
    log_success "GKEクラスタ '$GKE_CLUSTER_NAME' を作成しました"
fi

# kubectlの認証情報を取得
log_info "GKEクラスタの認証情報を取得中..."
gcloud container clusters get-credentials "$GKE_CLUSTER_NAME" --region="$GKE_ZONE"

# Kubernetes デプロイメントファイルの更新
log_info "Kubernetesデプロイメントファイルを更新中..."
sed "s/PROJECT_ID/$PROJECT_ID/g" k8s/deployment.yaml > k8s/deployment-updated.yaml

# GKEにデプロイ
log_info "AI推論サーバーをGKEにデプロイ中..."
kubectl apply -f k8s/deployment-updated.yaml

# サービスの外部IPを待機
log_info "Load Balancerの外部IPを待機中..."
echo "外部IPが割り当てられるまで数分かかる場合があります..."

EXTERNAL_IP=""
while [ -z "$EXTERNAL_IP" ]; do
    sleep 10
    EXTERNAL_IP=$(kubectl get service inference-server-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -z "$EXTERNAL_IP" ]; then
        log_info "外部IPを待機中... (30秒後に再試行)"
        sleep 20
    fi
done

log_success "AI推論サーバーの外部IP: $EXTERNAL_IP"
INFERENCE_SERVER_URL="http://$EXTERNAL_IP"

cd ../deploy

# ===== Phase 2: Webアプリケーションのデプロイ =====
log_info ""
log_info "=== Phase 2: Webアプリケーションのデプロイ ==="

# Docker イメージのビルド・プッシュ
log_info "WebアプリケーションのDockerイメージをビルド中..."
cd ../streamlit_app
gcloud builds submit --tag "$STREAMLIT_APP_IMAGE" .
log_success "WebアプリケーションのイメージをArtifact Registryにプッシュしました"

# Cloud Runにデプロイ
log_info "WebアプリケーションをCloud Runにデプロイ中..."
gcloud run deploy "$CLOUD_RUN_SERVICE_NAME" \
    --image "$STREAMLIT_APP_IMAGE" \
    --platform managed \
    --region "$CLOUD_RUN_REGION" \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars "INFERENCE_SERVER_URL=$INFERENCE_SERVER_URL/predict"

# Cloud RunのURLを取得
CLOUD_RUN_URL=$(gcloud run services describe "$CLOUD_RUN_SERVICE_NAME" \
    --region="$CLOUD_RUN_REGION" \
    --format="value(status.url)")

cd ../deploy

# ===== デプロイ完了 =====
log_success ""
log_success "🎉 デプロイが完了しました！"
echo ""
echo "=== デプロイ結果 ==="
echo "AI推論サーバー:"
echo "  - GKEクラスタ: $GKE_CLUSTER_NAME"
echo "  - 外部IP: $EXTERNAL_IP"
echo "  - エンドポイント: $INFERENCE_SERVER_URL/predict"
echo ""
echo "Webアプリケーション:"
echo "  - Cloud Runサービス: $CLOUD_RUN_SERVICE_NAME"
echo "  - URL: $CLOUD_RUN_URL"
echo ""
echo "=== アクセス方法 ==="
echo "1. Webアプリケーションにアクセス: $CLOUD_RUN_URL"
echo "2. 動画ファイルをアップロード"
echo "3. 検出したいオブジェクトを入力"
echo "4. 分析を実行"
echo ""
echo "=== 管理コマンド ==="
echo "# GKEクラスタの確認"
echo "kubectl get pods"
echo "kubectl get services"
echo ""
echo "# Cloud Runサービスの確認"
echo "gcloud run services list --region=$CLOUD_RUN_REGION"
echo ""
echo "# ログの確認"
echo "kubectl logs -l app=inference-server"
echo "gcloud run services logs read $CLOUD_RUN_SERVICE_NAME --region=$CLOUD_RUN_REGION"
echo ""

# 接続テスト
log_info "接続テストを実行中..."
if curl -f -s "$INFERENCE_SERVER_URL/health" >/dev/null; then
    log_success "✅ AI推論サーバーへの接続: OK"
else
    log_warning "⚠️ AI推論サーバーへの接続: 失敗（起動中の可能性があります）"
fi

if curl -f -s "$CLOUD_RUN_URL" >/dev/null; then
    log_success "✅ Webアプリケーションへの接続: OK"
else
    log_warning "⚠️ Webアプリケーションへの接続: 失敗（起動中の可能性があります）"
fi

log_success "デプロイスクリプトの実行が完了しました。上記のURLからアプリケーションにアクセスしてください。"