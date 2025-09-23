#!/usr/bin/env python3
"""
Flask版デバッグアプリケーション
Grounded-SAM-2 ビデオ分析システム（Streamlit UnicodeDecodeError回避版）
"""

import os
import sys
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_manager import VideoManager
    from video_tracker import create_tracking_video
except ImportError:
    print("Warning: video_manager or video_tracker modules not found")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
INFERENCE_API_URL = "http://localhost:8080"
COBOL_API_URL = "http://localhost:8081"
VIDEO_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

# Initialize VideoManager
try:
    video_manager = VideoManager("processed_videos")
except:
    video_manager = None

# Debug logs storage
debug_logs = []

def debug_log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    debug_logs.append(log_entry)
    print(log_entry)

# 高精度トラッキングシステムのチェック
ADVANCED_TRACKING_AVAILABLE = False
try:
    from yolo11_deepsort_sam2_tracker import create_advanced_tracking_video
    ADVANCED_TRACKING_AVAILABLE = True
    debug_log("✅ YOLO11+DeepSORT+SAM2システム利用可能", "SUCCESS")
except ImportError:
    debug_log("⚠️ 高精度システム利用不可、標準システムを使用", "WARNING")

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html', 
                         video_url=VIDEO_URL,
                         debug_logs=debug_logs[-50:],  # 最新50件のログ
                         archived_videos=get_archived_videos())

@app.route('/upload', methods=['POST'])
def upload_file():
    """ファイルアップロード処理"""
    debug_log("ファイルアップロード開始", "INFO")
    
    if 'file' not in request.files:
        debug_log("ファイルが選択されていません", "ERROR")
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '':
        debug_log("ファイル名が空です", "ERROR")
        return jsonify({'error': 'ファイル名が空です'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        debug_log(f"ファイル保存完了: {filename} ({os.path.getsize(filepath)} bytes)", "SUCCESS")
        
        # Process the video
        result = process_video_file(filepath)
        return jsonify(result)

@app.route('/process_url', methods=['POST'])
def process_url():
    """URL経由でビデオ処理"""
    data = request.get_json()
    url = data.get('url', VIDEO_URL)
    
    debug_log(f"URL経由でビデオ処理開始: {url}", "INFO")
    
    try:
        # Download video from URL with timeout
        debug_log("ビデオダウンロード開始...", "INFO")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        debug_log(f"ビデオダウンロード完了: {os.path.getsize(temp_file)} bytes", "SUCCESS")
        
        # Process the video
        result = process_video_file(temp_file)
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"URL処理エラー: {str(e)}"
        debug_log(error_msg, "ERROR")
        return jsonify({'error': error_msg}), 500

def process_video_file(video_path):
    """ビデオファイル処理"""
    debug_log(f"ビデオ処理開始: {video_path}", "INFO")
    
    try:
        # Check backend services
        inference_status = check_service_health(INFERENCE_API_URL)
        cobol_status = check_service_health(COBOL_API_URL)
        
        debug_log(f"Inference API状態: {inference_status}", "INFO")
        debug_log(f"COBOL API状態: {cobol_status}", "INFO")
        
        if not inference_status:
            return {'error': 'Inference APIサービスが利用できません'}
        
        # Call inference API
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {'text_prompt': 'person'}
            
            response = requests.post(f"{INFERENCE_API_URL}/predict", 
                                   files=files, data=data, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"Inference API エラー: {response.status_code}"
            debug_log(error_msg, "ERROR")
            return {'error': error_msg}
        
        result = response.json()
        debug_log("AI推論完了", "SUCCESS")
        
        # Call COBOL API if available
        if cobol_status:
            try:
                cobol_response = requests.post(f"{COBOL_API_URL}/legacy_process", 
                                             json=result, timeout=30)
                if cobol_response.status_code == 200:
                    cobol_result = cobol_response.json()
                    result['cobol_integration'] = cobol_result
                    debug_log("COBOL統合完了", "SUCCESS")
            except Exception as e:
                debug_log(f"COBOL統合エラー: {str(e)}", "WARNING")
        
        # Generate tracking video if video_tracker is available
        tracking_video_path = None
        try:
            from video_tracker import create_tracking_video
            tracking_result = create_tracking_video(result, video_path)
            debug_log(f"トラッキング動画生成完了: {tracking_result}", "SUCCESS")
            if tracking_result.get('success'):
                tracking_video_path = tracking_result.get('output_path')
        except Exception as e:
            debug_log(f"トラッキング動画生成エラー: {str(e)}", "ERROR")
        
        # Save result using VideoManager if available
        if video_manager and tracking_video_path:
            try:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                saved_success = video_manager.save_processing_result(
                    session_id, 
                    result.get('file_info', {}), 
                    tracking_video_path
                )
                if saved_success:
                    debug_log(f"結果保存完了: {session_id}", "SUCCESS")
                    result['session_id'] = session_id
                else:
                    debug_log("結果保存失敗", "ERROR")
            except Exception as e:
                debug_log(f"結果保存エラー: {str(e)}", "ERROR")
        
        # Add file info to result
        result['file_info'] = {
            'filename': os.path.basename(video_path),
            'size_mb': round(os.path.getsize(video_path) / (1024*1024), 2)
        }
        
        # Add processing statistics
        if 'boxes' in result:
            result['statistics'] = {
                'persons_detected': len(result['boxes']),
                'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return result
        
    except Exception as e:
        error_msg = f"ビデオ処理エラー: {str(e)}"
        debug_log(error_msg, "ERROR")
        return {'error': error_msg}

def check_service_health(url):
    """サービスヘルスチェック"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_archived_videos():
    """アーカイブされた動画一覧を取得"""
    if video_manager:
        try:
            return video_manager.list_all_videos()
        except Exception as e:
            debug_log(f"アーカイブ動画取得エラー: {str(e)}", "ERROR")
    return []

@app.route('/download/<session_id>')
def download_video(session_id):
    """動画ダウンロード"""
    if video_manager:
        try:
            result = video_manager.get_session_result(session_id)
            if result and 'persistent_video_path' in result:
                video_path = result['persistent_video_path']
                if os.path.exists(video_path):
                    return send_file(video_path, as_attachment=True)
        except Exception as e:
            debug_log(f"ダウンロードエラー: {str(e)}", "ERROR")
    return "Video not found", 404

@app.route('/logs')
def get_logs():
    """デバッグログ取得API"""
    return jsonify({'logs': debug_logs[-100:]})

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """ログクリア"""
    global debug_logs
    debug_logs = []
    debug_log("ログがクリアされました", "INFO")
    return jsonify({'status': 'success'})

@app.route('/test_process', methods=['POST'])
def test_process():
    """ローカルテストビデオで即座に処理テスト"""
    debug_log("ローカルテストビデオでの処理テスト開始", "INFO")
    
    try:
        # Use local test video
        test_video_path = os.path.join(os.getcwd(), 'test_sample.mp4')
        
        if not os.path.exists(test_video_path):
            debug_log(f"テストビデオファイルが見つかりません: {test_video_path}", "ERROR")
            return jsonify({'error': 'テストビデオファイルが見つかりません'}), 404
        
        debug_log(f"テストビデオファイル使用: {test_video_path} ({os.path.getsize(test_video_path)} bytes)", "INFO")
        
        # Process the video
        result = process_video_file(test_video_path)
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"テスト処理エラー: {str(e)}"
        debug_log(error_msg, "ERROR")
        return jsonify({'error': error_msg}), 500

@app.route('/advanced_test_process', methods=['POST'])
def advanced_test_process():
    """YOLO11+DeepSORT+SAM2高精度テスト処理"""
    debug_log("🚀 YOLO11+DeepSORT+SAM2高精度テスト開始", "INFO")
    
    try:
        if not ADVANCED_TRACKING_AVAILABLE:
            debug_log("高精度システムが利用できません", "WARNING")
            return jsonify({
                'error': '高精度システムが利用できません',
                'fallback': '標準テスト処理をお試しください'
            }), 503
        
        # Use local test video
        test_video_path = os.path.join(os.getcwd(), 'test_sample.mp4')
        
        if not os.path.exists(test_video_path):
            debug_log(f"テストビデオファイルが見つかりません: {test_video_path}", "ERROR")
            return jsonify({'error': 'テストビデオファイルが見つかりません'}), 404
        
        debug_log(f"🎯 高精度処理開始: {test_video_path}", "INFO")
        
        # 高精度トラッキング実行
        result = create_advanced_tracking_video(test_video_path, "processed_videos")
        
        if result['success']:
            debug_log("✅ YOLO11+DeepSORT+SAM2処理完了", "SUCCESS")
            
            # 結果を標準形式に変換
            processing_result = result['processing_result']
            
            response = {
                'success': True,
                'advanced_tracking': True,
                'output_video_path': result['output_path'],
                'processing_method': 'YOLO11+DeepSORT+SAM2',
                'stats': {
                    'total_frames': processing_result['total_frames_processed'],
                    'avg_fps': 1.0 / processing_result['final_stats']['avg_processing_time'] if processing_result['final_stats']['avg_processing_time'] > 0 else 0,
                    'total_detections': processing_result['final_stats']['total_detections'],
                    'total_tracks': processing_result['final_stats']['total_tracks']
                },
                'file_info': {
                    'filename': 'test_sample.mp4',
                    'size_mb': round(os.path.getsize(test_video_path) / (1024*1024), 2)
                },
                'technologies': ['YOLO11', 'DeepSORT', 'SAM2']
            }
            
            return jsonify(response)
        else:
            debug_log(f"❌ 高精度処理失敗: {result.get('error')}", "ERROR")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'advanced_tracking': False
            }), 500
        
    except Exception as e:
        error_msg = f"高精度テスト処理エラー: {str(e)}"
        debug_log(error_msg, "ERROR")
        return jsonify({'error': error_msg}), 500

@app.route('/test_advanced_server', methods=['POST'])
def test_advanced_server():
    """YOLO11+DeepSORT+SAM2サーバーテスト（リモート）"""
    debug_log("🌐 リモートYOLO11+DeepSORT+SAM2サーバーテスト開始", "INFO")
    
    try:
        # Advanced tracking server URL
        server_url = "https://5002-isr1fqyzouakmq5p9u4zf-6532622b.e2b.dev"
        
        # First test health endpoint
        health_response = requests.get(f"{server_url}/health", timeout=10)
        if health_response.status_code != 200:
            debug_log(f"サーバーヘルスチェック失敗: {health_response.status_code}", "ERROR")
            return jsonify({
                'error': 'Advanced tracking server not healthy',
                'status_code': health_response.status_code
            }), 503
        
        health_data = health_response.json()
        debug_log(f"✅ サーバー稼働確認: {health_data['status']}", "SUCCESS")
        
        # Test with sample image (create a test image)
        test_image_path = os.path.join(os.getcwd(), 'test_sample.mp4')
        
        if not os.path.exists(test_image_path):
            debug_log("テストファイルが見つかりません", "ERROR")
            return jsonify({'error': 'テストファイルが見つかりません'}), 404
        
        # Send test video to advanced tracking server
        debug_log("🎯 高精度トラッキングサーバーにリクエスト送信中...", "INFO")
        
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'video/mp4')}
            data = {
                'confidence': 0.3,
                'output_format': 'json'
            }
            
            response = requests.post(
                f"{server_url}/advanced_track",
                files=files,
                data=data,
                timeout=120  # 2 minutes timeout for video processing
            )
        
        if response.status_code == 200:
            result = response.json()
            debug_log("✅ リモート高精度トラッキング完了", "SUCCESS")
            
            return jsonify({
                'success': True,
                'remote_tracking': True,
                'server_url': server_url,
                'health_status': health_data,
                'tracking_result': result,
                'processing_method': 'Remote YOLO11+DeepSORT+SAM2',
                'technologies': ['YOLO11', 'DeepSORT', 'SAM2', 'Remote API']
            })
        else:
            debug_log(f"❌ リモートトラッキング失敗: {response.status_code}", "ERROR")
            return jsonify({
                'error': f'Remote tracking failed: {response.status_code}',
                'response_text': response.text[:500]  # First 500 chars
            }), response.status_code
            
    except requests.Timeout:
        debug_log("❌ リモートサーバータイムアウト", "ERROR")
        return jsonify({'error': 'Remote server timeout'}), 408
    except requests.ConnectionError:
        debug_log("❌ リモートサーバー接続エラー", "ERROR")
        return jsonify({'error': 'Cannot connect to remote server'}), 503
    except Exception as e:
        error_msg = f"リモートサーバーテストエラー: {str(e)}"
        debug_log(error_msg, "ERROR")
        return jsonify({'error': error_msg}), 500

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Grounded-SAM-2 デバッグアプリケーション (Flask版)</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: #ff4757;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .button {
            background: #ff4757;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .button:hover {
            background: #ff3742;
        }
        .button.secondary {
            background: #3742fa;
        }
        .button.secondary:hover {
            background: #2f32e2;
        }
        .log-container {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
        }
        .log-container div {
            margin-bottom: 2px;
        }
        .log-container .success-log { color: #2ecc71; }
        .log-container .error-log { color: #e74c3c; }
        .log-container .warning-log { color: #f39c12; }
        .log-container .info-log { color: #3498db; }
        .result-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .success { color: #27ae60; }
        .error { color: #e74c3c; }
        .warning { color: #f39c12; }
        .info { color: #3498db; }
        .file-input {
            margin: 10px 0;
            padding: 10px;
            border: 2px dashed #bdc3c7;
            border-radius: 5px;
            text-align: center;
        }
        .url-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            margin: 10px 0;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        .video-archive {
            max-height: 400px;
            overflow-y: auto;
        }
        .video-item {
            border: 1px solid #ddd;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Grounded-SAM-2 ビデオ分析システム</h1>
            <h2>Flask デバッグアプリケーション</h2>
            <p>人物検出・トラッキング・COBOL統合対応</p>
        </div>

        <div class="grid">
            <div>
                <div class="card">
                    <h3>📁 ファイルアップロード</h3>
                    <div class="file-input">
                        <input type="file" id="fileInput" accept="video/*,.mp4,.avi,.mov,.wmv,.mkv" />
                        <p>ビデオファイルを選択してください (MP4, AVI, MOV, WMV, MKV対応)</p>
                        <p id="fileInfo" style="color: #3498db; font-size: 12px; margin-top: 5px;"></p>
                    </div>
                    <button class="button" onclick="uploadFile()" id="uploadBtn" disabled>🚀 ファイルを処理</button>
                    <p style="font-size: 12px; color: #7f8c8d;">最大ファイルサイズ: 2GB</p>
                </div>

                <div class="card">
                    <h3>🔗 URL経由処理</h3>
                    <input type="text" class="url-input" id="urlInput" value="{{ video_url }}" placeholder="ビデオファイルのURLを入力">
                    <button class="button secondary" onclick="processUrl()">🌐 URLから処理開始</button>
                </div>

                <div class="card">
                    <h3>⚡ システム操作</h3>
                    <button class="button" onclick="testProcess()">🧪 ローカルテスト処理</button>
                    <button class="button" onclick="advancedTestProcess()" style="background: linear-gradient(45deg, #ff4757, #3742fa); color: white; font-weight: bold;">🚀 YOLO11+DeepSORT+SAM2</button>
                    <button class="button" onclick="testAdvancedServer()" style="background: linear-gradient(45deg, #2ecc71, #3498db); color: white; font-weight: bold;">🌐 リモート高精度サーバー</button>
                    <button class="button" onclick="refreshLogs()">🔄 ログ更新</button>
                    <button class="button" onclick="clearLogs()">🗑️ ログクリア</button>
                    <button class="button secondary" onclick="location.reload()">🔄 ページ再読み込み</button>
                </div>
            </div>

            <div>
                <div class="card">
                    <h3>📊 処理結果</h3>
                    <div class="loading" id="loading">
                        <p>🔄 処理中...</p>
                    </div>
                    <div id="result" class="result-container">
                        <p>処理結果がここに表示されます</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>📚 処理済み動画アーカイブ</h3>
            <div class="video-archive" id="videoArchive">
                {% for video in archived_videos %}
                <div class="video-item">
                    <strong>{{ video.timestamp }}</strong> - {{ video.session_id }} ({{ (video.file_size / (1024*1024)) | round(2) }} MB)
                    <button class="button secondary" onclick="downloadVideo('{{ video.session_id }}')">📥 ダウンロード</button>
                </div>
                {% endfor %}
                {% if not archived_videos %}
                <p>処理済み動画はありません</p>
                {% endif %}
            </div>
        </div>

        <div class="card">
            <h3>🔍 デバッグログ</h3>
            <div class="log-container" id="logContainer">
                {% for log in debug_logs %}
                <div>{{ log }}</div>
                {% endfor %}
            </div>
        </div>
    </div>

    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').innerHTML = '<p>処理中...</p>';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('ファイルを選択してください');
                return;
            }

            // ファイルサイズチェック (2GB = 2147483648 bytes)
            if (file.size > 2147483648) {
                alert('ファイルサイズが大きすぎます。2GB以下のファイルを選択してください。');
                return;
            }

            showLoading();
            document.getElementById('result').innerHTML = '<p class="info">📤 ファイルアップロード中: ' + file.name + ' (' + (file.size / (1024*1024)).toFixed(2) + ' MB)</p>';
            
            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                hideLoading();
                displayResult(data);
                refreshLogs();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p class="error">❌ アップロードエラー: ' + error + '</p>';
                refreshLogs();
            });
        }

        // ファイル選択時の処理
        function setupFileInput() {
            const fileInput = document.getElementById('fileInput');
            const fileInfo = document.getElementById('fileInfo');
            const uploadBtn = document.getElementById('uploadBtn');

            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const sizeMB = (file.size / (1024*1024)).toFixed(2);
                    fileInfo.innerHTML = '✅ 選択済み: ' + file.name + ' (' + sizeMB + ' MB)';
                    uploadBtn.disabled = false;
                    uploadBtn.style.opacity = '1';
                } else {
                    fileInfo.innerHTML = '';
                    uploadBtn.disabled = true;
                    uploadBtn.style.opacity = '0.5';
                }
            });
        }

        function processUrl() {
            const url = document.getElementById('urlInput').value;
            
            if (!url) {
                alert('URLを入力してください');
                return;
            }

            showLoading();

            fetch('/process_url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({url: url})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResult(data);
                refreshLogs();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p class="error">エラー: ' + error + '</p>';
            });
        }

        function testProcess() {
            showLoading();
            
            fetch('/test_process', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResult(data);
                refreshLogs();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p class="error">エラー: ' + error + '</p>';
            });
        }

        function advancedTestProcess() {
            showLoading();
            document.getElementById('result').innerHTML = '<p class="info">🚀 YOLO11+DeepSORT+SAM2高精度処理を開始中...</p>';
            
            fetch('/advanced_test_process', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayAdvancedResult(data);
                refreshLogs();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p class="error">❌ 高精度処理エラー: ' + error + '</p>';
            });
        }

        function testAdvancedServer() {
            showLoading();
            document.getElementById('result').innerHTML = '<p class="info">🌐 リモート高精度サーバーをテスト中...</p>';
            
            fetch('/test_advanced_server', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayRemoteServerResult(data);
                refreshLogs();
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p class="error">❌ リモートサーバーテストエラー: ' + error + '</p>';
            });
        }

        function displayRemoteServerResult(data) {
            const resultDiv = document.getElementById('result');
            
            if (data.error) {
                resultDiv.innerHTML = '<p class="error">❌ ' + data.error + '</p>';
                return;
            }

            if (!data.success) {
                resultDiv.innerHTML = '<p class="error">❌ リモートサーバーテスト失敗</p>';
                return;
            }

            let html = '<h4 class="success">🌐 リモート高精度サーバーテスト完了</h4>';
            
            if (data.server_url) {
                html += `<p><strong>🔗 サーバーURL:</strong> <a href="${data.server_url}" target="_blank">${data.server_url}</a></p>`;
            }
            
            if (data.health_status) {
                html += `<p><strong>💚 サーバーステータス:</strong> ${data.health_status.status}</p>`;
                html += `<p><strong>⏱️ レスポンス時間:</strong> ${data.health_status.timestamp}</p>`;
            }
            
            if (data.processing_method) {
                html += `<p><strong>🧮 処理方式:</strong> ${data.processing_method}</p>`;
            }
            
            if (data.technologies && Array.isArray(data.technologies)) {
                html += `<p><strong>🔧 利用技術:</strong> ${data.technologies.join(', ')}</p>`;
            }
            
            if (data.tracking_result && data.tracking_result.success) {
                const result = data.tracking_result;
                html += '<div class="stats">';
                html += '<h5>📊 処理結果統計</h5>';
                
                if (result.stats) {
                    html += `<p>📹 処理フレーム数: ${result.stats.frames_processed || 'N/A'}</p>`;
                    html += `<p>🎯 検出総数: ${result.stats.total_objects_detected || 0}</p>`;
                    html += `<p>🔍 追跡ID数: ${result.stats.unique_tracks || 0}</p>`;
                }
                
                if (result.processing_time) {
                    html += `<p>⏱️ 処理時間: ${result.processing_time.toFixed(2)}秒</p>`;
                }
                
                html += '</div>';
            }
            
            resultDiv.innerHTML = html;
        }

        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            
            if (data.error) {
                resultDiv.innerHTML = '<p class="error">❌ ' + data.error + '</p>';
                return;
            }

            let html = '<h4 class="success">✅ 処理完了</h4>';
            
            if (data.file_info) {
                html += '<p><strong>ファイル:</strong> ' + data.file_info.filename + ' (' + data.file_info.size_mb + ' MB)</p>';
            }
            
            if (data.statistics) {
                html += '<p><strong>検出された人物数:</strong> ' + data.statistics.persons_detected + '</p>';
                html += '<p><strong>処理時間:</strong> ' + data.statistics.processing_time + '</p>';
            }
            
            if (data.boxes && data.boxes.length > 0) {
                html += '<h5>🎯 検出結果:</h5>';
                html += '<ul>';
                for (let i = 0; i < data.boxes.length; i++) {
                    const box = data.boxes[i];
                    html += '<li>Person ' + (i+1) + ': [' + box.map(x => Math.round(x)).join(', ') + ']</li>';
                }
                html += '</ul>';
            }
            
            if (data.cobol_integration) {
                html += '<p class="success">🔗 COBOL統合: 完了</p>';
            }
            
            if (data.saved_info) {
                html += '<p class="success">💾 結果保存: ' + data.saved_info.unique_id + '</p>';
            }

            html += '<details><summary>📋 詳細結果 (JSON)</summary><pre>' + JSON.stringify(data, null, 2) + '</pre></details>';
            
            resultDiv.innerHTML = html;
        }

        function displayAdvancedResult(data) {
            const resultDiv = document.getElementById('result');
            
            if (data.error) {
                resultDiv.innerHTML = '<p class="error">❌ ' + data.error + '</p>';
                if (data.fallback) {
                    resultDiv.innerHTML += '<p class="warning">💡 ' + data.fallback + '</p>';
                }
                return;
            }

            let html = '<h4 class="success">🚀 YOLO11+DeepSORT+SAM2 処理完了</h4>';
            
            if (data.file_info) {
                html += '<p><strong>ファイル:</strong> ' + data.file_info.filename + ' (' + data.file_info.size_mb + ' MB)</p>';
            }
            
            if (data.advanced_tracking) {
                html += '<div style="background: linear-gradient(45deg, #2ecc71, #3498db); color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">';
                html += '<h5>🎯 高精度トラッキング結果</h5>';
                
                if (data.technologies) {
                    html += '<p><strong>使用技術:</strong> ' + data.technologies.join(' + ') + '</p>';
                }
                
                if (data.stats) {
                    html += '<p><strong>処理フレーム数:</strong> ' + data.stats.total_frames + '</p>';
                    html += '<p><strong>総検出数:</strong> ' + data.stats.total_detections + '</p>';
                    html += '<p><strong>追跡ID数:</strong> ' + data.stats.total_tracks + '</p>';
                    html += '<p><strong>平均FPS:</strong> ' + data.stats.avg_fps.toFixed(1) + '</p>';
                }
                
                if (data.output_video_path) {
                    html += '<p><strong>出力動画:</strong> ' + data.output_video_path + '</p>';
                }
                
                html += '</div>';
            }

            html += '<details><summary>📋 詳細結果 (JSON)</summary><pre>' + JSON.stringify(data, null, 2) + '</pre></details>';
            
            resultDiv.innerHTML = html;
        }

        function refreshLogs() {
            fetch('/logs')
            .then(response => response.json())
            .then(data => {
                const logContainer = document.getElementById('logContainer');
                logContainer.innerHTML = data.logs.map(log => {
                    let className = '';
                    if (log.includes('[SUCCESS]')) className = 'success-log';
                    else if (log.includes('[ERROR]')) className = 'error-log';
                    else if (log.includes('[WARNING]')) className = 'warning-log';
                    else if (log.includes('[INFO]')) className = 'info-log';
                    
                    return '<div class="' + className + '">' + log + '</div>';
                }).join('');
                logContainer.scrollTop = logContainer.scrollHeight;
            })
            .catch(error => {
                console.error('Error refreshing logs:', error);
            });
        }

        function clearLogs() {
            fetch('/clear_logs', {method: 'POST'})
            .then(() => {
                refreshLogs();
            })
            .catch(error => {
                console.error('Error clearing logs:', error);
            });
        }

        function downloadVideo(sessionId) {
            window.open('/download/' + sessionId, '_blank');
        }

        // Auto-refresh logs every 5 seconds
        setInterval(refreshLogs, 5000);

        // Initial setup
        document.addEventListener('DOMContentLoaded', function() {
            setupFileInput();
            refreshLogs();
        });
    </script>
</body>
</html>
"""

# Create templates directory and template
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    debug_log("Flask デバッグアプリケーション開始", "INFO")
    debug_log("Grounded-SAM-2 ビデオ分析システム 準備完了", "SUCCESS")
    
    # Check backend services
    inference_status = check_service_health(INFERENCE_API_URL)
    cobol_status = check_service_health(COBOL_API_URL)
    
    debug_log(f"Inference API ({INFERENCE_API_URL}): {'✅ 正常' if inference_status else '❌ 停止'}", 
              "SUCCESS" if inference_status else "WARNING")
    debug_log(f"COBOL API ({COBOL_API_URL}): {'✅ 正常' if cobol_status else '❌ 停止'}", 
              "SUCCESS" if cobol_status else "WARNING")
    
    app.run(host='0.0.0.0', port=8507, debug=True)