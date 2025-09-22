#!/bin/bash

# ローカル開発環境セットアップスクリプト
# このスクリプトは、ローカルでの開発・テスト用にアプリケーションを起動します。

set -e

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Python仮想環境の確認・作成
setup_python_env() {
    local dir=$1
    local env_name=$2
    
    log_info "$dir用のPython環境をセットアップ中..."
    
    cd "../$dir"
    
    # 仮想環境が存在しない場合は作成
    if [ ! -d "venv" ]; then
        log_info "仮想環境を作成中..."
        python3 -m venv venv
    fi
    
    # 仮想環境をアクティベート
    source venv/bin/activate
    
    # 依存関係をインストール
    log_info "依存関係をインストール中..."
    pip install -r requirements.txt
    
    log_success "$dir の環境セットアップが完了しました"
    
    cd ../deploy
}

# MOCKサーバーの作成
create_mock_cobol_server() {
    log_info "COBOL APIモックサーバーを作成中..."
    
    cat > "../mock_cobol_server.py" << 'EOF'
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# サンプルの製品データ
PRODUCTS = {
    "P001": {"product_name": "産業用ロボットアーム", "category": "machinery"},
    "P002": {"product_name": "自動車エンジン", "category": "automotive"},
    "P003": {"product_name": "スマートフォン", "category": "electronics"},
    "P004": {"product_name": "ノートパソコン", "category": "electronics"},
    "P005": {"product_name": "製造装置", "category": "machinery"}
}

@app.route('/get_product_info', methods=['GET'])
def get_product_info():
    product_id = request.args.get('id')
    
    if not product_id:
        return jsonify({'error': '製品IDが指定されていません'}), 400
    
    product = PRODUCTS.get(product_id)
    
    if not product:
        return jsonify({'error': '製品が見つかりません'}), 404
    
    return jsonify({
        'product_id': product_id,
        'product_name': product['product_name'],
        'category': product['category'],
        'status': 'success'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'cobol-mock-server',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'COBOL API Mock Server',
        'version': '1.0.0',
        'products': list(PRODUCTS.keys()),
        'endpoints': [
            '/get_product_info?id=P001 (GET) - 製品情報取得',
            '/health (GET) - ヘルスチェック'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))
EOF

    log_success "COBOLモックサーバーを作成しました"
}

# PM2設定ファイルの作成
create_pm2_config() {
    log_info "PM2設定ファイルを作成中..."
    
    cat > "../ecosystem.config.js" << 'EOF'
module.exports = {
  apps: [
    {
      name: 'inference-server',
      script: 'python3',
      args: 'inference_server/inference_server.py',
      cwd: '/home/user/webapp',
      env: {
        PORT: 8080,
        PYTHONPATH: '/home/user/webapp/inference_server'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: 'logs/inference-server-error.log',
      out_file: 'logs/inference-server-out.log',
      log_file: 'logs/inference-server.log'
    },
    {
      name: 'cobol-mock-server',
      script: 'python3',
      args: 'mock_cobol_server.py',
      cwd: '/home/user/webapp',
      env: {
        PORT: 8081
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      error_file: 'logs/cobol-mock-error.log',
      out_file: 'logs/cobol-mock-out.log',
      log_file: 'logs/cobol-mock.log'
    }
  ]
};
EOF

    # ログディレクトリを作成
    mkdir -p ../logs
    
    log_success "PM2設定ファイルを作成しました"
}

# Supervisor設定ファイルの作成
create_supervisor_config() {
    log_info "Supervisor設定ファイルを作成中..."
    
    cat > "../supervisord.conf" << 'EOF'
[supervisord]
nodaemon=false
logfile=/home/user/webapp/logs/supervisord.log
pidfile=/home/user/webapp/supervisord.pid

[unix_http_server]
file=/home/user/webapp/supervisor.sock

[supervisorctl]
serverurl=unix:///home/user/webapp/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[program:inference-server]
command=/home/user/webapp/inference_server/venv/bin/python /home/user/webapp/inference_server/inference_server.py
directory=/home/user/webapp/inference_server
autostart=true
autorestart=true
stdout_logfile=/home/user/webapp/logs/inference-server.log
stderr_logfile=/home/user/webapp/logs/inference-server-error.log
environment=PORT=8080

[program:cobol-mock-server]
command=python3 /home/user/webapp/mock_cobol_server.py
directory=/home/user/webapp
autostart=true
autorestart=true
stdout_logfile=/home/user/webapp/logs/cobol-mock.log
stderr_logfile=/home/user/webapp/logs/cobol-mock-error.log
environment=PORT=8081
EOF

    log_success "Supervisor設定ファイルを作成しました"
}

# メイン処理
main() {
    log_info "ローカル開発環境のセットアップを開始します..."
    
    # Python環境のセットアップ
    setup_python_env "inference_server" "inference"
    
    # Streamlit用に軽量セットアップ（venvは使わない）
    log_info "Streamlit用の依存関係をインストール中..."
    cd ../streamlit_app
    pip install --user -r requirements.txt
    cd ../deploy
    
    # モックサーバーとPM2設定の作成
    create_mock_cobol_server
    create_pm2_config
    create_supervisor_config
    
    # supervisorのインストール
    log_info "Supervisorをインストール中..."
    pip install --user supervisor
    
    log_success "ローカル開発環境のセットアップが完了しました！"
    
    echo ""
    echo "=== ローカル実行方法 ==="
    echo ""
    echo "# Option 1: Supervisorを使用（Python推奨）"
    echo "cd /home/user/webapp"
    echo "supervisord -c supervisord.conf"
    echo "supervisorctl -c supervisord.conf status"
    echo ""
    echo "# Option 2: 手動実行"
    echo "# ターミナル1: AI推論サーバー"
    echo "cd /home/user/webapp/inference_server && source venv/bin/activate && python inference_server.py"
    echo ""
    echo "# ターミナル2: COBOLモックサーバー"
    echo "cd /home/user/webapp && python mock_cobol_server.py"
    echo ""
    echo "# ターミナル3: Streamlitアプリケーション"
    echo "cd /home/user/webapp/streamlit_app"
    echo "INFERENCE_SERVER_URL=http://localhost:8080/predict COBOL_API_URL=http://localhost:8081/get_product_info streamlit run streamlit_app.py"
    echo ""
    echo "=== アクセスURL ==="
    echo "- Streamlitアプリ: http://localhost:8501"
    echo "- AI推論サーバー: http://localhost:8080"
    echo "- COBOLモックサーバー: http://localhost:8081"
    echo ""
    echo "=== テスト用製品ID ==="
    echo "P001, P002, P003, P004, P005"
}

# スクリプトの実行
main "$@"