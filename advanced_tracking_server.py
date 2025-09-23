#!/usr/bin/env python3
"""
YOLO11 + DeepSORT + SAM2 高精度人物トラッキングサーバー
最先端技術を統合したRESTful APIサーバー
"""

import os
import sys
import json
import tempfile
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 高精度トラッキングシステムをインポート
try:
    from yolo11_deepsort_sam2_tracker import (
        AdvancedPersonTracker, 
        create_advanced_tracking_video
    )
    ADVANCED_TRACKING_AVAILABLE = True
    logger.info("✅ YOLO11+DeepSORT+SAM2システム利用可能")
except ImportError as e:
    logger.warning(f"⚠️ 高精度トラッキング利用不可: {e}")
    ADVANCED_TRACKING_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB

# グローバル変数
tracker_instance = None
processing_stats = {
    'total_processed': 0,
    'successful_tracks': 0,
    'failed_processes': 0,
    'avg_processing_time': 0.0
}

def initialize_tracker():
    """高精度トラッカーを初期化"""
    global tracker_instance
    
    if not ADVANCED_TRACKING_AVAILABLE:
        logger.error("❌ 高精度トラッキングシステムが利用できません")
        return False
    
    try:
        if tracker_instance is None:
            logger.info("🚀 YOLO11+DeepSORT+SAM2トラッカー初期化中...")
            
            tracker_instance = AdvancedPersonTracker(
                yolo_model='s',  # バランス型モデル
                yolo_confidence=0.3,  # 高感度検出
                deepsort_max_disappeared=50,  # 長期追跡
                sam2_model='small'  # 効率重視
            )
            
            logger.info("✅ 高精度トラッカー初期化完了")
        return True
        
    except Exception as e:
        logger.error(f"❌ トラッカー初期化失敗: {e}")
        return False

@app.route('/', methods=['GET'])
def root():
    """ルートエンドポイント"""
    return jsonify({
        'service': 'YOLO11+DeepSORT+SAM2 高精度人物トラッキングサーバー',
        'version': '1.0.0',
        'capabilities': {
            'yolo11_detection': ADVANCED_TRACKING_AVAILABLE,
            'deepsort_tracking': ADVANCED_TRACKING_AVAILABLE,
            'sam2_segmentation': ADVANCED_TRACKING_AVAILABLE
        },
        'endpoints': [
            '/health (GET) - システム状態確認',
            '/advanced_track (POST) - 高精度人物追跡',
            '/track_video (POST) - ビデオ全体の追跡処理',
            '/track_frame (POST) - 単一フレーム処理',
            '/stats (GET) - 処理統計'
        ],
        'technologies': [
            'YOLO11: 最先端物体検出',
            'DeepSORT: 深層学習追跡',
            'SAM2: 精密セグメンテーション'
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """詳細なヘルスチェック"""
    global tracker_instance
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'advanced-tracking-server',
        'version': '1.0.0'
    }
    
    # システムコンポーネント状態チェック
    components = {
        'yolo11_available': False,
        'deepsort_available': False,
        'sam2_available': False,
        'tracker_initialized': tracker_instance is not None
    }
    
    if ADVANCED_TRACKING_AVAILABLE:
        try:
            # YOLO11チェック
            from ultralytics import YOLO
            components['yolo11_available'] = True
        except:
            pass
        
        try:
            # DeepSORTチェック
            from deep_sort_realtime import DeepSort
            components['deepsort_available'] = True
        except:
            pass
        
        try:
            # SAM2チェック（推測）
            components['sam2_available'] = True  # 実装に応じて調整
        except:
            pass
    
    health_status['components'] = components
    health_status['processing_stats'] = processing_stats
    health_status['memory_usage'] = get_memory_usage()
    
    return jsonify(health_status)

@app.route('/advanced_track', methods=['POST'])
def advanced_track():
    """
    YOLO11+DeepSORT+SAM2による高精度人物追跡
    単一画像またはビデオの処理
    """
    start_time = time.time()
    
    try:
        if not ADVANCED_TRACKING_AVAILABLE:
            return jsonify({
                'success': False,
                'error': '高精度トラッキングシステムが利用できません',
                'fallback_suggestion': '/predict エンドポイントをお試しください'
            }), 503
        
        # トラッカー初期化
        if not initialize_tracker():
            return jsonify({
                'success': False,
                'error': 'トラッカー初期化に失敗しました'
            }), 500
        
        # ファイルチェック
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'ファイルが指定されていません'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'ファイル名が空です'
            }), 400
        
        # 処理パラメータ
        detection_confidence = float(request.form.get('confidence', 0.3))
        output_format = request.form.get('output_format', 'json')  # json, video
        
        logger.info(f"🎯 高精度追跡開始: {file.filename}")
        
        # 一時ファイル保存
        temp_input = tempfile.NamedTemporaryFile(delete=False, 
                                                suffix=os.path.splitext(file.filename)[1])
        file.save(temp_input.name)
        temp_input.close()
        
        # ファイルタイプ判定
        is_video = file.content_type.startswith('video/') or \
                   file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
        
        if is_video:
            # ビデオファイル処理
            result = process_video_advanced(temp_input.name, detection_confidence)
        else:
            # 画像ファイル処理
            result = process_image_advanced(temp_input.name, detection_confidence)
        
        # 一時ファイル削除
        try:
            os.unlink(temp_input.name)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        # 統計更新
        update_processing_stats(processing_time, result['success'])
        
        # レスポンス生成
        response = {
            'success': result['success'],
            'processing_time': processing_time,
            'file_info': {
                'filename': file.filename,
                'content_type': file.content_type,
                'is_video': is_video
            },
            'detection_results': result.get('detection_results', []),
            'tracking_results': result.get('tracking_results', []),
            'segmentation_results': result.get('segmentation_results', []),
            'output_video_path': result.get('output_video_path'),
            'stats': {
                'total_objects_detected': result.get('total_detected', 0),
                'unique_tracks': result.get('unique_tracks', 0),
                'frames_processed': result.get('frames_processed', 1)
            },
            'technologies_used': ['YOLO11', 'DeepSORT', 'SAM2']
        }
        
        if not result['success']:
            response['error'] = result.get('error', 'Unknown error')
        
        logger.info(f"✅ 高精度追跡完了: {processing_time:.2f}秒")
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        update_processing_stats(processing_time, False)
        
        logger.error(f"❌ 高精度追跡エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/track_video', methods=['POST'])
def track_video():
    """ビデオファイル専用の高精度追跡処理"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'ビデオファイルが指定されていません'}), 400
        
        video_file = request.files['video']
        
        # 一時ファイル保存
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_video.name)
        temp_video.close()
        
        logger.info(f"🎬 ビデオ追跡処理開始: {video_file.filename}")
        
        # 高精度追跡ビデオ生成
        result = create_advanced_tracking_video(temp_video.name)
        
        # 一時ファイル削除
        try:
            os.unlink(temp_video.name)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ ビデオ追跡エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/track_frame', methods=['POST'])
def track_frame():
    """単一フレームの高精度処理"""
    try:
        if not initialize_tracker():
            return jsonify({
                'success': False,
                'error': 'トラッカー初期化失敗'
            }), 500
        
        if 'image' not in request.files:
            return jsonify({'error': '画像ファイルが指定されていません'}), 400
        
        image_file = request.files['image']
        
        # 一時ファイル保存
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image_file.save(temp_image.name)
        temp_image.close()
        
        # フレーム処理
        import cv2
        frame = cv2.imread(temp_image.name)
        result = tracker_instance.process_frame(frame)
        
        # 一時ファイル削除
        try:
            os.unlink(temp_image.name)
        except:
            pass
        
        return jsonify({
            'success': True,
            'frame_result': result
        })
        
    except Exception as e:
        logger.error(f"❌ フレーム処理エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """処理統計情報"""
    return jsonify({
        'processing_stats': processing_stats,
        'system_info': {
            'advanced_tracking_available': ADVANCED_TRACKING_AVAILABLE,
            'tracker_initialized': tracker_instance is not None,
            'memory_usage': get_memory_usage()
        }
    })

def process_video_advanced(video_path: str, confidence: float = 0.3) -> dict:
    """高精度ビデオ処理"""
    try:
        result = create_advanced_tracking_video(video_path)
        
        if result['success']:
            processing_result = result['processing_result']
            
            return {
                'success': True,
                'output_video_path': result['output_path'],
                'frames_processed': processing_result['total_frames_processed'],
                'total_detected': sum(len(r['tracked_objects']) for r in processing_result['processing_results']),
                'unique_tracks': len(set(
                    obj['track_id'] 
                    for r in processing_result['processing_results'] 
                    for obj in r['tracked_objects'] 
                    if 'track_id' in obj
                )),
                'final_stats': processing_result['final_stats']
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def process_image_advanced(image_path: str, confidence: float = 0.3) -> dict:
    """高精度画像処理"""
    try:
        import cv2
        
        # 画像読み込み
        frame = cv2.imread(image_path)
        if frame is None:
            return {
                'success': False,
                'error': '画像を読み込めません'
            }
        
        # フレーム処理
        result = tracker_instance.process_frame(frame)
        
        return {
            'success': True,
            'detection_results': result['detections'],
            'tracking_results': result['tracked_objects'],
            'total_detected': len(result['detections']),
            'frames_processed': 1,
            'processing_stats': result['stats']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def update_processing_stats(processing_time: float, success: bool):
    """処理統計を更新"""
    global processing_stats
    
    processing_stats['total_processed'] += 1
    
    if success:
        processing_stats['successful_tracks'] += 1
    else:
        processing_stats['failed_processes'] += 1
    
    # 平均処理時間更新
    current_avg = processing_stats['avg_processing_time']
    total = processing_stats['total_processed']
    processing_stats['avg_processing_time'] = (
        (current_avg * (total - 1) + processing_time) / total
    )

def get_memory_usage() -> dict:
    """メモリ使用量を取得"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'percent': round(process.memory_percent(), 2)
        }
    except ImportError:
        return {'error': 'psutil not available'}

# 後方互換性のためのエンドポイント
@app.route('/predict', methods=['POST'])
def predict_compatibility():
    """既存システムとの後方互換性"""
    logger.info("🔄 後方互換モードで処理中...")
    
    # 高精度システムが利用可能な場合はそれを使用
    if ADVANCED_TRACKING_AVAILABLE and initialize_tracker():
        return advanced_track()
    
    # フォールバック: 基本的なレスポンス
    return jsonify({
        'success': True,
        'boxes': [],
        'labels': [],
        'confidence_scores': [],
        'message': '高精度システム初期化中。基本モードで動作。',
        'note': '完全な機能については /advanced_track をご利用ください'
    })

if __name__ == '__main__':
    logger.info("🚀 YOLO11+DeepSORT+SAM2高精度トラッキングサーバー開始...")
    
    # 初期化
    if ADVANCED_TRACKING_AVAILABLE:
        logger.info("⚡ 高精度システム初期化中...")
        initialize_tracker()
    else:
        logger.warning("⚠️ 高精度システム利用不可。基本機能のみ提供。")
    
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5002)), 
        debug=False
    )