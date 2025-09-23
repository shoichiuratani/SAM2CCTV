import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import json
import time
from datetime import datetime
import tempfile

app = Flask(__name__)

# OpenCVのHOG人物検出器を初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("🤖 OpenCV HOG人物検出器を初期化...")
print("✅ 実際の人物検出機能が利用可能です")

def detect_persons_in_image(image_path):
    """OpenCVを使用した実際の人物検出"""
    try:
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        height, width = image.shape[:2]
        
        # HOG人物検出を実行
        (rects, weights) = hog.detectMultiScale(
            image,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.0
        )
        
        # 検出結果を処理
        persons = []
        for i, ((x, y, w, h), weight) in enumerate(zip(rects, weights)):
            # 正規化座標に変換
            x1_norm = x / width
            y1_norm = y / height
            x2_norm = (x + w) / width
            y2_norm = (y + h) / height
            
            # 信頼度を0-1の範囲に正規化
            confidence = min(max(weight[0] * 0.5 + 0.5, 0.0), 1.0)
            
            persons.append({
                'id': i + 1,
                'bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                'bbox_pixels': [x, y, x + w, y + h],
                'confidence': float(confidence),
                'detection_method': 'OpenCV_HOG'
            })
        
        return {
            'success': True,
            'persons_detected': len(persons),
            'persons': persons,
            'image_dimensions': {'width': width, 'height': height}
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'persons_detected': 0,
            'persons': []
        }

def detect_persons_in_video(video_path, max_frames=5):
    """ビデオファイル内の人物検出"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"ビデオファイルを開けません: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 処理するフレーム間隔を計算
        frame_interval = max(1, total_frames // max_frames)
        
        all_persons = []
        processed_frames = 0
        
        frame_number = 0
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 指定間隔でフレームを処理
            if frame_number % frame_interval == 0:
                height, width = frame.shape[:2]
                
                # 一時ファイルに保存
                temp_image_path = f"/tmp/frame_{frame_number}.jpg"
                cv2.imwrite(temp_image_path, frame)
                
                # 人物検出を実行
                detection_result = detect_persons_in_image(temp_image_path)
                
                if detection_result['success'] and detection_result['persons']:
                    timestamp = frame_number / fps if fps > 0 else 0
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    milliseconds = int((timestamp % 1) * 1000)
                    
                    all_persons.append({
                        'frame_number': frame_number,
                        'timestamp': f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}",
                        'persons': detection_result['persons']
                    })
                
                processed_frames += 1
                
                # 一時ファイルを削除
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            
            frame_number += 1
        
        cap.release()
        
        return {
            'success': True,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'persons_detected': all_persons
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'persons_detected': []
        }

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """ビデオファイルの実際の人物検出分析"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'ビデオファイルが指定されていません'}), 400

        video_file = request.files['video']
        task = request.form.get('task', 'person_detection')
        
        # 一時ファイルに保存
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_video.name)
        temp_video.close()
        
        # ファイル情報の取得
        file_size = os.path.getsize(temp_video.name)
        
        print(f"🔍 ビデオ人物検出開始: {video_file.filename} ({file_size} bytes)")
        
        # 実際の人物検出を実行
        detection_results = detect_persons_in_video(temp_video.name)
        
        # 一時ファイルを削除
        try:
            os.remove(temp_video.name)
        except:
            pass
        
        response = {
            'success': detection_results['success'],
            'task': task,
            'video_info': {
                'filename': video_file.filename,
                'size_bytes': file_size,
                'format': video_file.content_type
            },
            'detection_results': detection_results,
            'processing_time': time.time(),
            'model_version': 'OpenCV-HOG-Person-Detection',
            'processed_at': datetime.now().isoformat()
        }
        
        total_persons = sum(len(frame['persons']) for frame in detection_results.get('persons_detected', []))
        print(f"✅ 検出完了: {total_persons} 人検出")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ ビデオ検出エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """画像の実際の人物検出"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルが指定されていません'}), 400

        file = request.files['file']
        text_prompt = request.form.get('text_prompt', 'person')
        
        # 一時ファイルに保存
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_image.name)
        temp_image.close()
        
        print(f"🔍 画像人物検出開始: {file.filename}")
        
        # 実際の人物検出を実行
        detection_result = detect_persons_in_image(temp_image.name)
        
        # 一時ファイルを削除
        try:
            os.remove(temp_image.name)
        except:
            pass
        
        if detection_result['success'] and detection_result['persons']:
            # 検出結果をFlaskアプリで期待される形式に変換
            boxes = []
            labels = []
            confidence_scores = []
            
            for person in detection_result['persons']:
                boxes.append(person['bbox'])
                labels.append('person')
                confidence_scores.append(person['confidence'])
            
            response = {
                'success': True,
                'boxes': boxes,
                'labels': labels,
                'confidence_scores': confidence_scores,
                'detection_method': 'OpenCV_HOG',
                'total_detected': len(boxes)
            }
            
            print(f"✅ 検出完了: {len(boxes)} 人検出")
            
        else:
            # 人物が検出されない場合
            response = {
                'success': True,
                'boxes': [],
                'labels': [],
                'confidence_scores': [],
                'detection_method': 'OpenCV_HOG',
                'total_detected': 0,
                'note': '人物が検出されませんでした'
            }
            
            print("⚠️ 人物検出なし")

        return jsonify(response)
        
    except Exception as e:
        print(f"❌ 画像検出エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'boxes': [],
            'labels': [],
            'confidence_scores': []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'healthy',
        'service': 'inference-server',
        'version': '1.0.0',
        'model': 'OpenCV-HOG-Person-Detection',
        'uptime': time.time(),
        'endpoints': ['/analyze_video', '/predict', '/health'],
        'detection_capability': 'Real person detection using OpenCV'
    })

@app.route('/', methods=['GET'])
def root():
    """ルートエンドポイント"""
    return jsonify({
        'message': 'OpenCV-HOG 人物検出サーバー',
        'version': '1.0.0',
        'endpoints': [
            '/analyze_video (POST) - ビデオ人物検出',
            '/predict (POST) - 画像人物検出', 
            '/health (GET) - ヘルスチェック'
        ],
        'detection_method': 'OpenCV HOG (Histogram of Oriented Gradients)',
        'note': 'この版は実際の人物検出機能を提供します'
    })

if __name__ == '__main__':
    print("🤖 OpenCV HOG人物検出サーバーを開始...")
    print("🔍 実際の人物検出機能が有効です")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)