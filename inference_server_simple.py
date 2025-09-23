import os
from flask import Flask, request, jsonify
import json
import time
from datetime import datetime

app = Flask(__name__)

# シミュレーション用のAIモデル初期化
print("AIモデルのロード処理を初期化（シミュレーション）")

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """ビデオファイルの人物検出分析"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'ビデオファイルが指定されていません'}), 400

        video_file = request.files['video']
        task = request.form.get('task', 'person_detection')
        
        # ファイル情報の取得
        file_size = len(video_file.read())
        video_file.seek(0)  # ファイルポインタをリセット
        
        # シミュレーション処理（実際のGrounded-SAM-2処理の代わり）
        time.sleep(2)  # 処理時間のシミュレーション
        
        # 模擬的な人物検出結果
        mock_results = {
            'success': True,
            'task': task,
            'video_info': {
                'filename': video_file.filename,
                'size_bytes': file_size,
                'format': video_file.content_type
            },
            'detection_results': {
                'total_frames': 150,
                'frames_with_persons': 142,
                'persons_detected': [
                    {
                        'frame_number': 1,
                        'timestamp': '00:00:01',
                        'persons': [
                            {'id': 1, 'confidence': 0.95, 'bbox': [100, 150, 200, 350]},
                            {'id': 2, 'confidence': 0.87, 'bbox': [300, 120, 400, 380]}
                        ]
                    },
                    {
                        'frame_number': 30,
                        'timestamp': '00:00:30', 
                        'persons': [
                            {'id': 1, 'confidence': 0.92, 'bbox': [110, 160, 210, 360]}
                        ]
                    }
                ]
            },
            'processing_time': 2.1,
            'model_version': 'Grounded-SAM-2-mock',
            'processed_at': datetime.now().isoformat()
        }
        
        return jsonify(mock_results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """画像の物体検出（後方互換性のため）"""
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが指定されていません'}), 400

    file = request.files['file']
    text_prompt = request.form.get('prompt', 'person')

    # シミュレーションとして、画像の中心に検出ボックスを返します
    mock_boxes = [[0.25, 0.25, 0.75, 0.75]]  # 正規化座標 [x1, y1, x2, y2]
    mock_labels = [text_prompt]

    response = {
        'success': True,
        'boxes': mock_boxes,
        'labels': mock_labels,
        'confidence_scores': [0.95]
    }

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'healthy',
        'service': 'inference-server',
        'version': '1.0.0',
        'model': 'Grounded-SAM-2-simulation',
        'uptime': time.time(),
        'endpoints': ['/analyze_video', '/predict', '/health']
    })

@app.route('/', methods=['GET'])
def root():
    """ルートエンドポイント"""
    return jsonify({
        'message': 'Grounded-SAM-2 Inference Server (Simulation Mode)',
        'version': '1.0.0',
        'endpoints': [
            '/analyze_video (POST) - ビデオ人物検出',
            '/predict (POST) - 画像物体検出', 
            '/health (GET) - ヘルスチェック'
        ],
        'note': 'この版はシミュレーションモードで動作しています'
    })

if __name__ == '__main__':
    print("🤖 Grounded-SAM-2 推論サーバー（シミュレーションモード）を開始...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)