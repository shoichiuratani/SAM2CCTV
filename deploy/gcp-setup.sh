#!/bin/bash

# Google Cloud Platform セットアップスクリプト
# このスクリプトは、Grounded-SAM-2動画分析アプリケーションのデプロイに必要なGCPリソースを設定します。

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

# プロジェクトIDの確認
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    log_error "GCPプロジェクトが設定されていません。"
    log_info "以下のコマンドでプロジェクトを設定してください："
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

log_info "使用するGCPプロジェクト: $PROJECT_ID"

# 必要なAPIの有効化
log_info "必要なGoogle Cloud APIを有効化しています..."
APIS=(
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "container.googleapis.com"
    "run.googleapis.com"
    "compute.googleapis.com"
)

for api in "${APIS[@]}"; do
    log_info "APIを有効化中: $api"
    gcloud services enable "$api"
done

log_success "すべてのAPIが有効化されました"

# Artifact Registryの設定
log_info "Artifact Registryリポジトリを作成しています..."
REPOSITORY_NAME="grounded-sam2-repo"
REGION="asia-northeast1"

# リポジトリが既に存在するかチェック
if gcloud artifacts repositories describe "$REPOSITORY_NAME" --location="$REGION" >/dev/null 2>&1; then
    log_warning "Artifact Registryリポジトリ '$REPOSITORY_NAME' は既に存在します"
else
    gcloud artifacts repositories create "$REPOSITORY_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Grounded-SAM-2 Video Analysis Application Repository"
    log_success "Artifact Registryリポジトリ '$REPOSITORY_NAME' を作成しました"
fi

# Docker認証の設定
log_info "Docker認証を設定しています..."
gcloud auth configure-docker "$REGION-docker.pkg.dev"

# 必要な権限の確認
log_info "必要な権限を確認しています..."
REQUIRED_ROLES=(
    "roles/container.developer"
    "roles/run.developer"
    "roles/cloudbuild.builds.editor"
    "roles/artifactregistry.writer"
)

CURRENT_USER=$(gcloud config get-value account)
for role in "${REQUIRED_ROLES[@]}"; do
    log_info "権限を確認中: $role"
    # 権限チェック（エラーを無視）
    gcloud projects get-iam-policy "$PROJECT_ID" --flatten="bindings[].members" \
        --format="table(bindings.role)" --filter="bindings.members:$CURRENT_USER AND bindings.role:$role" >/dev/null 2>&1 || true
done

# 環境変数ファイルの作成
log_info "環境変数ファイルを作成しています..."
cat > "../.env" << EOF
# Google Cloud Platform 設定
PROJECT_ID=$PROJECT_ID
REGION=$REGION
REPOSITORY_NAME=$REPOSITORY_NAME
ARTIFACT_REGISTRY_URL=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME

# Docker イメージ名
INFERENCE_SERVER_IMAGE=\${ARTIFACT_REGISTRY_URL}/inference-server:v1
STREAMLIT_APP_IMAGE=\${ARTIFACT_REGISTRY_URL}/streamlit-app:v1

# GKE設定
GKE_CLUSTER_NAME=ai-cluster
GKE_ZONE=$REGION

# Cloud Run設定
CLOUD_RUN_SERVICE_NAME=streamlit-app
CLOUD_RUN_REGION=$REGION
EOF

log_success "環境変数ファイル '.env' を作成しました"

# 設定情報の表示
log_success "GCPセットアップが完了しました！"
echo ""
echo "=== 設定情報 ==="
echo "プロジェクトID: $PROJECT_ID"
echo "リージョン: $REGION"
echo "Artifact Registry: $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME"
echo ""
echo "=== 次の手順 ==="
echo "1. ./deploy-all.sh を実行してアプリケーションをデプロイ"
echo "2. または個別にコンポーネントをデプロイ"
echo ""
log_info "デプロイスクリプトを実行する準備ができました"