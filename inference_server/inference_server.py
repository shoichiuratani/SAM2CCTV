import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# 注意: ここではGrounded-SAM-2のセットアップをシミュレートしています。
# 実際のデプロイでは、公式リポジトリからモデルのロード処理を実装する必要があります。
print("AIモデルのロード処理を初期化（シミュレーション）")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが指定されていません'}), 400

    file = request.files['file']
    text_prompt = request.form.get('prompt', '')

    if not text_prompt:
        return jsonify({'error': '検出プロンプトがありません'}), 400

    # ファイルを読み込み、OpenCVで扱える形式に変換
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- ここからAIによる推論処理（シミュレーション） ---
    # 本来はこの部分でGrounded-SAM-2の推論が実行されます。
    # シミュレーションとして、画像の中心に検出ボックスを返します。
    h, w, _ = img.shape
    boxes = [[w*0.25, h*0.25, w*0.75, h*0.75]] # 形式: [x1, y1, x2, y2]
    labels = [text_prompt]

    response = {
        'boxes': boxes,
        'labels': labels
    }

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'healthy',
        'service': 'inference-server',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def root():
    """ルートエンドポイント"""
    return jsonify({
        'message': 'Grounded-SAM-2 Inference Server',
        'version': '1.0.0',
        'endpoints': [
            '/predict (POST) - AI推論実行',
            '/health (GET) - ヘルスチェック'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))