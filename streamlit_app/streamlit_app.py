import streamlit as st
import requests
import tempfile
import cv2
import numpy as np
import os

# 環境変数からAPIサーバーのURLを取得
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8080/predict")
COBOL_API_URL = os.getenv("COBOL_API_URL", "http://localhost:8081/get_product_info")

st.set_page_config(
    page_title="Grounded-SAM-2 動画分析アプリケーション",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Grounded-SAM-2 動画分析アプリケーション 🚀")

# サイドバーで設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # APIサーバーの接続確認
    st.subheader("📡 API接続状況")
    try:
        # AI推論サーバーのヘルスチェック
        inference_health_url = INFERENCE_SERVER_URL.replace("/predict", "/health")
        response = requests.get(inference_health_url, timeout=5)
        if response.ok:
            st.success("✅ AI推論サーバー: 接続OK")
        else:
            st.error("❌ AI推論サーバー: 接続エラー")
    except requests.exceptions.RequestException:
        st.error("❌ AI推論サーバー: 接続失敗")
    
    # COBOL APIの接続確認
    try:
        response = requests.get(COBOL_API_URL.replace("/get_product_info", "/health"), timeout=5)
        if response.ok:
            st.success("✅ COBOL API: 接続OK")
        else:
            st.warning("⚠️ COBOL API: 接続不可（オプション）")
    except requests.exceptions.RequestException:
        st.warning("⚠️ COBOL API: 接続失敗（オプション）")
    
    st.divider()
    
    # 詳細設定
    st.subheader("🔧 詳細設定")
    confidence_threshold = st.slider("信頼度閾値", 0.0, 1.0, 0.5, 0.1)
    max_video_duration = st.slider("最大動画長（秒）", 10, 300, 60, 10)

# メインコンテンツ
col1, col2 = st.columns(2)

with col1:
    st.header("1. 入力設定")
    
    # COBOL連携セクション
    st.subheader("🏢 COBOL連携（オプション）")
    product_id = st.text_input("製品ID", help="基幹システムから製品情報を取得します")
    
    # 製品IDが入力されたらCOBOL APIを叩いて製品名を取得し、プロンプトの初期値に設定
    cobol_product_name = ""
    if product_id:
        try:
            response = requests.get(f"{COBOL_API_URL}?id={product_id}", timeout=10)
            if response.ok:
                product_data = response.json()
                cobol_product_name = product_data.get("product_name", "")
                if cobol_product_name:
                    st.success(f"✅ 製品名取得: {cobol_product_name}")
                else:
                    st.warning("⚠️ 製品名が見つかりません")
        except requests.exceptions.RequestException:
            st.warning("⚠️ COBOL APIへの接続に失敗しました。手動で入力してください。")
    
    st.divider()
    
    # 動画アップロード
    st.subheader("📹 動画アップロード")
    uploaded_file = st.file_uploader(
        "動画ファイルを選択", 
        type=["mp4", "mov", "avi"],
        help="対応形式: MP4, MOV, AVI"
    )
    
    if uploaded_file:
        st.success(f"✅ ファイル選択: {uploaded_file.name}")
        # 動画の基本情報を表示
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 ファイルサイズ: {file_size:.1f} MB")
    
    # 検出プロンプト
    st.subheader("🎯 検出設定")
    prompt_text = st.text_input(
        "検出したいオブジェクト名", 
        value=cobol_product_name,
        help="AI分析で検出したいオブジェクトを入力してください"
    )
    
    # 分析実行ボタン
    st.divider()
    run_button = st.button(
        "🚀 分析を実行", 
        type="primary", 
        use_container_width=True,
        disabled=not (uploaded_file and prompt_text)
    )
    
    if not uploaded_file:
        st.info("👆 動画ファイルを選択してください")
    elif not prompt_text:
        st.info("👆 検出したいオブジェクト名を入力してください")

with col2:
    st.header("2. 分析結果")
    
    result_placeholder = st.empty()
    
    # 初期状態
    if not run_button:
        with result_placeholder.container():
            st.info("🎬 ここに分析結果の動画が表示されます")
            st.markdown("""
            **分析手順:**
            1. 動画ファイルをアップロード
            2. 検出したいオブジェクト名を入力
            3. 「分析を実行」ボタンをクリック
            4. AI分析結果が動画に重畳されて表示されます
            """)

# 分析実行処理
if run_button and uploaded_file is not None and prompt_text:
    with result_placeholder.container():
        with st.spinner("🤖 AIサーバーと通信し、動画を分析中です..."):
            try:
                # 動画を一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_in:
                    t_in.write(uploaded_file.read())
                    temp_video_path = t_in.name

                cap = cv2.VideoCapture(temp_video_path)
                
                # 動画の基本情報を取得
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.info(f"📊 動画情報: {width}x{height}, {fps:.1f}fps, {duration:.1f}秒, {total_frames}フレーム")
                
                if duration > max_video_duration:
                    st.warning(f"⚠️ 動画が長すぎます（{duration:.1f}秒 > {max_video_duration}秒）。最初の{max_video_duration}秒のみ処理します。")

                # 1フレーム目を取得してAIサーバーに送信
                ret, first_frame = cap.read()
                if not ret:
                    st.error("❌ 動画フレームの読み込みに失敗しました。")
                else:
                    # 画像エンコード
                    _, img_encoded = cv2.imencode('.jpg', first_frame)
                    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                    data = {'prompt': prompt_text}

                    # AI推論APIに送信
                    response = requests.post(INFERENCE_SERVER_URL, files=files, data=data, timeout=30)
                    
                    if response.ok:
                        results = response.json()
                        st.success("✅ AIによる分析が完了しました。結果動画を生成します。")
                        
                        # プログレスバー
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 結果描画用の動画ファイルを設定
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_out:
                            output_video_path = t_out.name

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                        # 全フレームに検出結果を描画
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ポインタを最初に戻す
                        
                        frame_count = 0
                        max_frames = min(int(fps * max_video_duration), total_frames)
                        
                        while cap.isOpened() and frame_count < max_frames:
                            ret_frame, frame = cap.read()
                            if not ret_frame:
                                break

                            # 検出ボックスとラベルを描画
                            for box, label in zip(results.get('boxes', []), results.get('labels', [])):
                                x1, y1, x2, y2 = map(int, box)
                                # 境界チェック
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(width, x2), min(height, y2)
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            
                            out.write(frame)
                            frame_count += 1
                            
                            # プログレスバーを更新
                            progress = frame_count / max_frames
                            progress_bar.progress(progress)
                            status_text.text(f"フレーム処理中: {frame_count}/{max_frames}")

                        out.release()
                        progress_bar.progress(1.0)
                        status_text.text("✅ 動画生成完了")
                        
                        # 結果動画を表示
                        st.success("🎉 分析完了！検出結果が動画に重畳されました。")
                        st.video(output_video_path)
                        
                        # 検出結果の詳細
                        st.subheader("📋 検出結果詳細")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("検出オブジェクト数", len(results.get('boxes', [])))
                        with col_b:
                            st.metric("処理フレーム数", frame_count)
                        
                        # 検出されたオブジェクトの一覧
                        if results.get('labels'):
                            st.write("**検出されたオブジェクト:**")
                            for i, (label, box) in enumerate(zip(results.get('labels', []), results.get('boxes', [])), 1):
                                x1, y1, x2, y2 = box
                                st.write(f"{i}. {label} - 位置: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
                        
                        # 一時ファイルのクリーンアップ
                        try:
                            os.unlink(temp_video_path)
                            os.unlink(output_video_path)
                        except:
                            pass
                            
                    else:
                        st.error(f"❌ AIサーバーでエラーが発生しました: {response.text}")
                        
            except requests.exceptions.RequestException as e:
                st.error(f"❌ AIサーバーへの接続に失敗しました: {e}")
            except Exception as e:
                st.error(f"❌ 予期しないエラーが発生しました: {e}")
            finally:
                # リソースのクリーンアップ
                if 'cap' in locals():
                    cap.release()

# フッター
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by Grounded-SAM-2 | 🏢 COBOL Legacy Integration | ☁️ Google Cloud Platform
</div>
""", unsafe_allow_html=True)