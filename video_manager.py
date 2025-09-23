import os
import json
import shutil
from datetime import datetime

class VideoManager:
    """動画ファイルとセッション状態の永続化管理クラス"""
    
    def __init__(self, base_dir="/home/user/webapp/processed_videos"):
        self.base_dir = base_dir
        self.results_file = os.path.join(base_dir, "results.json")
        self.ensure_directories()
    
    def ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_processing_result(self, session_id, file_info, tracking_video_path=None):
        """処理結果を永続化保存"""
        try:
            # 既存の結果を読み込み
            results = self.load_all_results()
            
            # セッションIDに基づいて結果を保存
            result_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'file_info': file_info,
                'tracking_video_path': tracking_video_path,
                'status': 'completed'
            }
            
            # 動画ファイルを永続ディレクトリにコピー
            if tracking_video_path and os.path.exists(tracking_video_path):
                filename = f"tracking_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                persistent_path = os.path.join(self.base_dir, filename)
                shutil.copy2(tracking_video_path, persistent_path)
                result_data['persistent_video_path'] = persistent_path
                print(f"📁 動画を永続化保存: {persistent_path}")
            
            results[session_id] = result_data
            
            # 結果をファイルに保存
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 処理結果を永続化保存: session_id={session_id}")
            return True
            
        except Exception as e:
            print(f"❌ 処理結果保存エラー: {e}")
            return False
    
    def load_all_results(self):
        """全ての処理結果を読み込み"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"❌ 結果読み込みエラー: {e}")
            return {}
    
    def get_session_result(self, session_id):
        """特定セッションの結果を取得"""
        results = self.load_all_results()
        return results.get(session_id)
    
    def list_all_videos(self):
        """全ての処理済み動画を一覧表示"""
        results = self.load_all_results()
        video_list = []
        
        for session_id, data in results.items():
            if data.get('persistent_video_path') and os.path.exists(data['persistent_video_path']):
                video_info = {
                    'session_id': session_id,
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'video_path': data['persistent_video_path'],
                    'file_size': os.path.getsize(data['persistent_video_path']),
                    'file_info': data.get('file_info', {})
                }
                video_list.append(video_info)
        
        # タイムスタンプ順にソート
        video_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return video_list
    
    def create_session_id(self):
        """ユニークなセッションIDを作成"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def cleanup_old_files(self, days=7):
        """古いファイルをクリーンアップ"""
        try:
            current_time = datetime.now()
            results = self.load_all_results()
            cleaned_sessions = []
            
            for session_id, data in list(results.items()):
                try:
                    timestamp = datetime.fromisoformat(data.get('timestamp', ''))
                    age = (current_time - timestamp).days
                    
                    if age > days:
                        # 古いファイルを削除
                        if data.get('persistent_video_path') and os.path.exists(data['persistent_video_path']):
                            os.remove(data['persistent_video_path'])
                        
                        # 結果から削除
                        del results[session_id]
                        cleaned_sessions.append(session_id)
                        
                except Exception as e:
                    print(f"⚠️ セッション {session_id} のクリーンアップでエラー: {e}")
            
            # 更新された結果を保存
            if cleaned_sessions:
                with open(self.results_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"🧹 {len(cleaned_sessions)} 個の古いセッションをクリーンアップしました")
            
        except Exception as e:
            print(f"❌ クリーンアップエラー: {e}")

# グローバルインスタンス
video_manager = VideoManager()

# Flask専用バージョン - Streamlit依存削除