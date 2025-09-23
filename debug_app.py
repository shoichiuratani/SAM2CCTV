import streamlit as st
import sys
import traceback
import os
import tempfile
import time
import requests
from datetime import datetime
import pandas as pd
import numpy as np

# デバッグモード設定
DEBUG_MODE = True

# API設定
INFERENCE_API_URL = "http://localhost:8080"
COBOL_API_URL = "http://localhost:8081"

# ページ設定
st.set_page_config(
    page_title="🐛 デバッグ版 Grounded-SAM-2 人物検出システム",
    page_icon="🐛",
    layout="wide"
)

# デバッグ用CSS
st.markdown("""
<style>
.debug-header {
    background: linear-gradient(90deg, #ff4757 0%, #ff6b81 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
    font-family: monospace;
    white-space: pre-wrap;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.debug-log {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 10px;
    border-radius: 5px;
    font-family: monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def debug_log(message, level="INFO"):
    """デバッグログ記録関数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG_MODE:
        if 'debug_logs' not in st.session_state:
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(f"[{timestamp}] [{level}] {message}")
        # 最新1000件のログのみ保持
        if len(st.session_state.debug_logs) > 1000:
            st.session_state.debug_logs = st.session_state.debug_logs[-1000:]

def safe_execute(func, description="操作"):
    """安全な実行ラッパー"""
    try:
        debug_log(f"{description}を開始します", "INFO")
        result = func()
        debug_log(f"{description}が正常に完了しました", "SUCCESS")
        return result
    except Exception as e:
        error_msg = f"{description}でエラーが発生: {str(e)}"
        debug_log(error_msg, "ERROR")
        debug_log(f"スタックトレース:\n{traceback.format_exc()}", "DEBUG")
        st.error(f"❌ {error_msg}")
        return None

def check_api_health():
    """API サーバーのヘルスチェック"""
    results = {}
    
    # Inference Server チェック
    try:
        response = requests.get(f"{INFERENCE_API_URL}/health", timeout=5)
        if response.status_code == 200:
            results['inference'] = {"status": "healthy", "data": response.json()}
            debug_log("推論サーバーは正常です", "SUCCESS")
        else:
            results['inference'] = {"status": "error", "message": f"HTTP {response.status_code}"}
            debug_log(f"推論サーバーエラー: HTTP {response.status_code}", "WARNING")
    except Exception as e:
        results['inference'] = {"status": "error", "message": str(e)}
        debug_log(f"推論サーバーに接続できません: {e}", "ERROR")
    
    # COBOL Server チェック
    try:
        response = requests.get(f"{COBOL_API_URL}/health", timeout=5)
        if response.status_code == 200:
            results['cobol'] = {"status": "healthy", "data": response.json()}
            debug_log("COBOLサーバーは正常です", "SUCCESS")
        else:
            results['cobol'] = {"status": "error", "message": f"HTTP {response.status_code}"}
            debug_log(f"COBOLサーバーエラー: HTTP {response.status_code}", "WARNING")
    except Exception as e:
        results['cobol'] = {"status": "error", "message": str(e)}
        debug_log(f"COBOLサーバーに接続できません: {e}", "ERROR")
    
    return results

def init_session_state():
    """セッション状態の初期化"""
    debug_log("セッション状態の初期化を開始", "INFO")
    
    # 永続化マネージャーをインポート
    try:
        from video_manager import restore_session_data, get_or_create_session_id
        
        # セッションIDを取得または作成
        session_id = get_or_create_session_id()
        debug_log(f"セッションID: {session_id}", "INFO")
        
        # 保存されたデータを復元
        restored = restore_session_data()
        if restored:
            debug_log("保存されたセッションデータを復元しました", "SUCCESS")
    except Exception as e:
        debug_log(f"セッション復元エラー: {e}", "WARNING")
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    if 'uploaded_file_info' not in st.session_state:
        st.session_state.uploaded_file_info = None
        
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
        
    if 'system_diagnostics' not in st.session_state:
        st.session_state.system_diagnostics = None
    
    debug_log("セッション状態を初期化しました", "SUCCESS")

def get_system_diagnostics():
    """システム診断情報の取得"""
    try:
        diagnostics = {
            "Python Version": sys.version,
            "Streamlit Version": st.__version__,
            "Working Directory": os.getcwd(),
            "Available Memory": f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3):.2f} GB",
            "Session State Keys": list(st.session_state.keys()),
            "Temporary Directory": tempfile.gettempdir()
        }
        debug_log("システム診断情報を取得しました", "SUCCESS")
        return diagnostics
    except Exception as e:
        debug_log(f"システム診断情報取得エラー: {e}", "ERROR")
        return {"error": str(e)}

def process_video_from_url(video_url):
    """URL経由でのビデオファイル処理"""
    try:
        debug_log(f"URL経由でファイルを取得中: {video_url}", "INFO")
        
        # URLからファイルをダウンロード
        response = requests.get(video_url, timeout=30)
        if response.status_code != 200:
            debug_log(f"ファイル取得エラー: HTTP {response.status_code}", "ERROR")
            return None
            
        file_content = response.content
        file_size = len(file_content)
        
        debug_log(f"ファイルサイズ: {file_size} bytes", "INFO")
        
        # 一時ファイルに保存
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
            
        debug_log(f"一時ファイル保存: {temp_path}", "SUCCESS")
        
        # 推論サーバーへの送信
        try:
            with open(temp_path, 'rb') as f:
                files = {'video': ('video.mp4', f, 'video/mp4')}
                debug_log("推論サーバーにファイルを送信中...", "INFO")
                
                api_response = requests.post(
                    f"{INFERENCE_API_URL}/analyze_video",
                    files=files,
                    data={'task': 'person_detection'},
                    timeout=60
                )
                
                if api_response.status_code == 200:
                    result = api_response.json()
                    debug_log("推論サーバーからの応答を受信", "SUCCESS")
                    debug_log(f"推論結果: {result}", "INFO")
                    
                    return {
                        "name": "video.mp4",
                        "size": file_size,
                        "type": "video/mp4",
                        "upload_time": datetime.now().isoformat(),
                        "temp_path": temp_path,
                        "inference_result": result,
                        "source": "url_download"
                    }
                else:
                    debug_log(f"推論サーバーエラー: HTTP {api_response.status_code}", "ERROR")
                    return None
                    
        except Exception as e:
            debug_log(f"推論サーバー通信エラー: {e}", "ERROR")
            return None
            
    except Exception as e:
        debug_log(f"URL処理エラー: {e}", "ERROR")
        return None

def process_video_upload(uploaded_file):
    """ビデオファイルのアップロード処理"""
    if uploaded_file is None:
        debug_log("アップロードファイルがありません", "WARNING")
        return None
    
    debug_log(f"ファイルアップロード開始: {uploaded_file.name}", "INFO")
    debug_log(f"ファイルサイズ: {uploaded_file.size} bytes", "INFO")
    debug_log(f"ファイルタイプ: {uploaded_file.type}", "INFO")
    
    # ファイル情報の保存
    file_info = {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "type": uploaded_file.type,
        "upload_time": datetime.now().isoformat()
    }
    
    # 一時ファイルに保存
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            debug_log(f"一時ファイル作成: {tmp_file.name}", "INFO")
            tmp_file.write(uploaded_file.getbuffer())
            file_info["temp_path"] = tmp_file.name
            
        debug_log("ファイルを一時領域に保存しました", "SUCCESS")
        
        # 推論サーバーへの送信テスト
        try:
            with open(file_info["temp_path"], 'rb') as f:
                files = {'video': (uploaded_file.name, f, uploaded_file.type)}
                debug_log("推論サーバーにファイルを送信中...", "INFO")
                
                response = requests.post(
                    f"{INFERENCE_API_URL}/analyze_video",
                    files=files,
                    data={'task': 'person_detection'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    debug_log("推論サーバーからの応答を受信", "SUCCESS")
                    debug_log(f"推論結果: {result}", "INFO")
                    file_info["inference_result"] = result
                else:
                    debug_log(f"推論サーバーエラー: HTTP {response.status_code}", "ERROR")
                    debug_log(f"エラー内容: {response.text}", "ERROR")
                    file_info["inference_error"] = f"HTTP {response.status_code}: {response.text}"
                    
        except Exception as e:
            debug_log(f"推論サーバー通信エラー: {e}", "ERROR")
            file_info["inference_error"] = str(e)
        
        return file_info
        
    except Exception as e:
        debug_log(f"ファイル処理エラー: {e}", "ERROR")
        return None

def main():
    """メインアプリケーション"""
    # セッション状態の初期化
    safe_execute(init_session_state, "セッション状態初期化")
    
    # ヘッダー
    st.markdown("""
    <div class="debug-header">
        <h1>🐛 デバッグ版 Grounded-SAM-2 人物検出システム</h1>
        <p>ファイルアップロード診断とエラー追跡システム</p>
    </div>
    """, unsafe_allow_html=True)
    
    # システム診断パネル
    with st.expander("🔍 システム診断情報", expanded=True):
        if st.button("診断情報を更新"):
            st.session_state.system_diagnostics = safe_execute(
                get_system_diagnostics, 
                "システム診断情報取得"
            )
        
        if st.session_state.system_diagnostics:
            for key, value in st.session_state.system_diagnostics.items():
                st.write(f"**{key}**: {value}")
    
    # API ヘルスチェック
    with st.expander("🏥 API サーバー状態", expanded=True):
        if st.button("ヘルスチェック実行"):
            health_results = safe_execute(check_api_health, "APIヘルスチェック")
            
            if health_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 推論サーバー")
                    if health_results['inference']['status'] == 'healthy':
                        st.success("✅ 正常動作中")
                        if 'data' in health_results['inference']:
                            st.json(health_results['inference']['data'])
                    else:
                        st.error(f"❌ エラー: {health_results['inference']['message']}")
                
                with col2:
                    st.subheader("💼 COBOLサーバー")
                    if health_results['cobol']['status'] == 'healthy':
                        st.success("✅ 正常動作中")
                        if 'data' in health_results['cobol']:
                            st.json(health_results['cobol']['data'])
                    else:
                        st.error(f"❌ エラー: {health_results['cobol']['message']}")
    
    st.divider()
    
    # ファイルアップロード
    st.header("📁 ビデオファイルアップロード")
    
    debug_log("ファイルアップローダーを作成します", "INFO")
    uploaded_file = st.file_uploader(
        "ビデオファイルを選択してください",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="対応形式: MP4, AVI, MOV, MKV, WebM (最大1GB)"
    )
    debug_log("ファイルアップローダーを作成しました", "SUCCESS")
    
    # ファイル状態の詳細デバッグ
    debug_log(f"アップロードファイル状態チェック: uploaded_file = {uploaded_file}", "DEBUG")
    debug_log(f"uploaded_file is None: {uploaded_file is None}", "DEBUG")
    debug_log(f"uploaded_file type: {type(uploaded_file)}", "DEBUG")
    
    if uploaded_file is not None:
        debug_log(f"ファイルが選択されました: {uploaded_file.name}", "SUCCESS")
        debug_log(f"ファイルサイズ: {uploaded_file.size} bytes", "INFO")
        debug_log(f"ファイルタイプ: {uploaded_file.type}", "INFO")
        
        # 成功メッセージ表示
        st.success(f"✅ ファイルが正常にアップロードされました: {uploaded_file.name}")
        
        # ファイル情報表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ファイル名", uploaded_file.name)
        with col2:
            st.metric("サイズ", f"{uploaded_file.size / (1024*1024):.2f} MB")
        with col3:
            st.metric("形式", uploaded_file.type)
        
        # 処理ボタン
        debug_log("ファイル処理ボタンを表示します", "INFO")
        if st.button("🚀 ビデオを処理", type="primary"):
            debug_log("ファイル処理ボタンがクリックされました", "SUCCESS")
            with st.spinner("ファイルを処理中..."):
                result = safe_execute(
                    lambda: process_video_upload(uploaded_file),
                    "ビデオファイル処理"
                )
                
                if result:
                    st.session_state.uploaded_file_info = result
                    st.success("✅ ファイル処理が完了しました！")
                    
                    # 結果表示
                    with st.expander("📊 処理結果詳細", expanded=True):
                        st.json(result)
    else:
        # ファイルが選択されていない場合の診断情報
        debug_log("ファイルが選択されていません (uploaded_file is None)", "WARNING")
        st.info("📝 ファイルが選択されていません。上記のファイルアップローダーでビデオファイルを選択してください。")
        
        # 代替手段: URL経由でのファイル処理
        st.divider()
        st.header("🔗 代替手段: URL経由でビデオを処理")
        st.info("ファイルアップロードに問題がある場合は、ビデオファイルのURLを直接入力して処理できます。")
        
        # URLが提供されている場合の自動処理
        provided_url = "https://page.gensparksite.com/get_upload_url/a4e735154ab381559e373d17ff6c77f652d1997f504527efee5e92e049f871a0/default/b64ed6ee-aef5-41a3-8abf-8845b04cb206"
        
        st.write("**提供されたファイル情報:**")
        st.write(f"- ファイル名: test.mp4")
        st.write(f"- サイズ: 48,170,507 bytes (約46MB)")
        st.write(f"- MIMEタイプ: video/mp4")
        
        # URL処理ボタン
        if st.button("🚀 提供されたURLで処理を開始", type="primary"):
            debug_log(f"URL処理を開始: {provided_url}", "INFO")
            with st.spinner("URLからファイルを取得して処理中..."):
                result = safe_execute(
                    lambda: process_video_from_url(provided_url),
                    "URL経由ビデオファイル処理"
                )
                
                if result:
                    st.session_state.uploaded_file_info = result
                    st.success("✅ URL経由でのファイル処理が完了しました！")
                    
                    # 結果表示
                    with st.expander("📊 処理結果詳細", expanded=True):
                        st.json(result)
                else:
                    st.error("❌ URL経由での処理に失敗しました。デバッグログをご確認ください。")
        
        # 詳細な診断情報表示
        with st.expander("🔍 ファイルアップロード診断", expanded=False):
            st.write("**現在のセッション状態:**")
            if hasattr(st.session_state, 'keys'):
                for key in st.session_state.keys():
                    if 'upload' in key.lower() or 'file' in key.lower():
                        st.write(f"- {key}: {getattr(st.session_state, key, 'N/A')}")
            
            st.write("**期待される動作:**")
            st.write("1. ファイルを選択すると `uploaded_file` オブジェクトが作成される")
            st.write("2. ファイル情報（名前、サイズ、形式）が表示される") 
            st.write("3. 「🚀 ビデオを処理」ボタンが表示される")
    
    # 処理結果表示セクション
    if 'uploaded_file_info' in st.session_state and st.session_state.uploaded_file_info:
        st.divider()
        st.header("🎉 処理結果")
        
        file_info = st.session_state.uploaded_file_info
        
        # ファイル情報表示
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ファイル名", file_info.get('name', 'Unknown'))
        with col2:
            st.metric("ファイルサイズ", f"{file_info.get('size', 0) / (1024*1024):.1f} MB")
        with col3:
            st.metric("処理方法", file_info.get('source', 'Unknown'))
        with col4:
            st.metric("処理時間", f"{file_info.get('inference_result', {}).get('processing_time', 'N/A')} 秒")
        
        # 人物検出結果の詳細表示
        if 'inference_result' in file_info:
            result = file_info['inference_result']
            
            if result.get('success'):
                st.success("✅ 人物検出処理が正常に完了しました！")
                
                detection = result.get('detection_results', {})
                
                # 検出統計
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総フレーム数", detection.get('total_frames', 0))
                with col2:
                    st.metric("人物検出フレーム", detection.get('frames_with_persons', 0))
                with col3:
                    detection_rate = 0
                    if detection.get('total_frames', 0) > 0:
                        detection_rate = (detection.get('frames_with_persons', 0) / detection.get('total_frames', 1)) * 100
                    st.metric("検出率", f"{detection_rate:.1f}%")
                
                # 検出詳細
                persons_data = detection.get('persons_detected', [])
                if persons_data:
                    st.subheader("🎯 検出された人物の詳細")
                    
                    for i, frame_data in enumerate(persons_data[:5]):  # 最初の5フレームを表示
                        with st.expander(f"フレーム {frame_data.get('frame_number', i+1)} - {frame_data.get('timestamp', 'N/A')}", expanded=i==0):
                            persons = frame_data.get('persons', [])
                            
                            if persons:
                                # 人物ごとの情報をテーブル表示
                                import pandas as pd
                                
                                person_data = []
                                for person in persons:
                                    bbox = person.get('bbox', [0, 0, 0, 0])
                                    person_data.append({
                                        'ID': person.get('id', 'N/A'),
                                        '信頼度': f"{person.get('confidence', 0):.2f}",
                                        'X座標': f"{bbox[0]}-{bbox[2]}",
                                        'Y座標': f"{bbox[1]}-{bbox[3]}",
                                        'サイズ': f"{bbox[2]-bbox[0]} x {bbox[3]-bbox[1]}"
                                    })
                                
                                df = pd.DataFrame(person_data)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("このフレームでは人物が検出されませんでした")
                
                # モデル情報
                st.subheader("🤖 AI モデル情報")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**モデル**: {result.get('model_version', 'Unknown')}")
                    st.write(f"**タスク**: {result.get('task', 'Unknown')}")
                with col2:
                    st.write(f"**処理日時**: {result.get('processed_at', 'Unknown')}")
                    st.write(f"**処理時間**: {result.get('processing_time', 'Unknown')} 秒")
                
                # ROIトラッキング動画生成
                st.divider()
                st.subheader("🎬 ROIトラッキング動画生成")
                st.info("検出結果をもとに、人物のROI（関心領域）とIDトラッキングを含む視覚化動画を生成します。")
                
                if st.button("🎯 ROIトラッキング動画を生成", type="primary"):
                    with st.spinner("ROIトラッキング動画を生成中..."):
                        try:
                            # video_tracker モジュールをインポート
                            from video_tracker import create_tracking_video
                            
                            # 一時ファイルパスを取得
                            temp_video_path = file_info.get('temp_path')
                            if not temp_video_path:
                                st.error("❌ 元動画ファイルが見つかりません")
                            else:
                                debug_log(f"ROIトラッキング動画生成開始: {temp_video_path}", "INFO")
                                
                                # トラッキング動画を生成
                                tracking_result = create_tracking_video(result, temp_video_path)
                                
                                if tracking_result.get('success'):
                                    output_path = tracking_result['output_path']
                                    video_info = tracking_result['video_info']
                                    
                                    debug_log(f"ROIトラッキング動画生成完了: {output_path}", "SUCCESS")
                                    
                                    # 永続化保存
                                    try:
                                        from video_manager import video_manager, get_or_create_session_id
                                        session_id = get_or_create_session_id()
                                        success = video_manager.save_processing_result(
                                            session_id, 
                                            file_info, 
                                            output_path
                                        )
                                        if success:
                                            debug_log("処理結果を永続化保存しました", "SUCCESS")
                                        else:
                                            debug_log("永続化保存に失敗しました", "WARNING")
                                    except Exception as e:
                                        debug_log(f"永続化保存エラー: {e}", "ERROR")
                                    
                                    st.success("✅ ROIトラッキング動画の生成が完了しました！")
                                    
                                    # 動画情報を表示
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("出力ファイルサイズ", f"{video_info['file_size_mb']:.1f} MB")
                                    with col2:
                                        st.metric("生成日時", video_info['created_at'][:19])
                                    
                                    # 動画プレビュー
                                    st.subheader("🎥 ROIトラッキング動画プレビュー")
                                    
                                    # 動画を読み込んで表示
                                    if os.path.exists(output_path):
                                        with open(output_path, 'rb') as video_file:
                                            video_bytes = video_file.read()
                                        
                                        st.video(video_bytes)
                                        
                                        # ダウンロードボタン
                                        st.download_button(
                                            label="📥 ROIトラッキング動画をダウンロード",
                                            data=video_bytes,
                                            file_name=f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                            mime="video/mp4"
                                        )
                                        
                                        # 処理詳細
                                        with st.expander("🔍 トラッキング処理詳細", expanded=False):
                                            st.write("**実装された機能:**")
                                            st.write("- 🎯 人物検出結果に基づくバウンディングボックス描画")
                                            st.write("- 🏷️ 各人物への個別ID割り当てと色分け表示")
                                            st.write("- 📊 リアルタイム信頼度スコア表示")
                                            st.write("- ⚡ フレーム間補間による滑らかな追跡")
                                            st.write("- 📝 タイムスタンプとフレーム情報のオーバーレイ")
                                            st.write("- 🎨 ROI内部の半透明ハイライト")
                                            st.write("- ✚ 中心点クロスハイヤー表示")
                                            
                                            st.json(tracking_result)
                                    else:
                                        st.error("❌ 生成された動画ファイルが見つかりません")
                                        
                                else:
                                    debug_log(f"ROIトラッキング動画生成失敗: {tracking_result.get('error', 'Unknown error')}", "ERROR")
                                    st.error(f"❌ ROIトラッキング動画の生成に失敗しました: {tracking_result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            debug_log(f"ROIトラッキング動画生成エラー: {e}", "ERROR")
                            st.error(f"❌ エラーが発生しました: {e}")
                
                # 詳細結果のJSON表示
                with st.expander("📋 完全な検出結果 (JSON)", expanded=False):
                    st.json(result)
            else:
                st.error("❌ 人物検出処理でエラーが発生しました")
                if 'error' in result:
                    st.error(f"エラー詳細: {result['error']}")
    
    # 動画アーカイブセクション
    st.divider()
    st.header("📚 処理済み動画アーカイブ")
    
    try:
        from video_manager import video_manager
        
        # クリーンアップボタン
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🧹 古いファイルをクリーンアップ"):
                video_manager.cleanup_old_files(days=1)
                st.rerun()
        
        # 全ての処理済み動画を表示
        video_list = video_manager.list_all_videos()
        
        if video_list:
            st.success(f"📁 {len(video_list)} 個の処理済み動画が保存されています")
            
            for i, video_info in enumerate(video_list):
                with st.expander(f"🎬 動画 {i+1} - {video_info['timestamp'][:19]}", expanded=(i==0)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("セッションID", video_info['session_id'][-8:])
                    with col2:
                        st.metric("ファイルサイズ", f"{video_info['file_size'] / (1024*1024):.1f} MB")
                    with col3:
                        st.metric("生成日時", video_info['timestamp'][:19])
                    
                    # 動画を表示
                    video_path = video_info['video_path']
                    if os.path.exists(video_path):
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        st.video(video_bytes)
                        
                        # ダウンロードボタン
                        st.download_button(
                            label=f"📥 動画 {i+1} をダウンロード",
                            data=video_bytes,
                            file_name=f"roi_tracking_{video_info['session_id']}.mp4",
                            mime="video/mp4",
                            key=f"download_{i}"
                        )
                    else:
                        st.error("❌ 動画ファイルが見つかりません")
        else:
            st.info("📝 まだ処理済み動画がありません。上記でビデオ処理を実行してください。")
    
    except Exception as e:
        st.error(f"❌ アーカイブ表示エラー: {e}")
        debug_log(f"アーカイブ表示エラー: {e}", "ERROR")
    
    st.divider()
    
    # デバッグログパネル
    st.header("🔧 リアルタイムデバッグログ")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📄 ログをクリア"):
            st.session_state.debug_logs = []
            debug_log("デバッグログをクリアしました", "SUCCESS")
    
    with col2:
        auto_refresh = st.checkbox("自動更新", value=True)
    
    with col3:
        log_level_filter = st.selectbox(
            "ログレベルフィルター",
            ["ALL", "SUCCESS", "INFO", "WARNING", "ERROR", "DEBUG"]
        )
    
    # ログ表示
    if 'debug_logs' in st.session_state and st.session_state.debug_logs:
        # フィルタリング
        filtered_logs = st.session_state.debug_logs
        if log_level_filter != "ALL":
            filtered_logs = [log for log in st.session_state.debug_logs if f"[{log_level_filter}]" in log]
        
        # 最新50件を表示
        recent_logs = filtered_logs[-50:] if len(filtered_logs) > 50 else filtered_logs
        
        st.markdown(
            f'<div class="debug-log">{"<br>".join(recent_logs)}</div>',
            unsafe_allow_html=True
        )
        
        # ログ統計
        st.caption(f"表示中: {len(recent_logs)} / 全体: {len(st.session_state.debug_logs)} ログエントリー")
        
        if auto_refresh:
            time.sleep(1)
            st.rerun()
    else:
        st.info("📝 デバッグログはまだありません")
    
    # フッター情報
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
        🐛 デバッグモード有効 | Grounded-SAM-2 人物検出システム v2.0<br>
        問題が発生した場合は、上記のデバッグログとシステム診断情報をご確認ください
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        debug_log("アプリケーション開始", "SUCCESS")
        main()
    except Exception as e:
        st.error(f"❌ アプリケーションエラー: {e}")
        st.code(traceback.format_exc())
        debug_log(f"アプリケーションエラー: {e}", "ERROR")