import cv2
import numpy as np
from datetime import datetime
import os
import json

class PersonTracker:
    """人物検出結果をもとにROIトラッキングを行うクラス"""
    
    def __init__(self, detection_results, video_path, output_path=None):
        """
        初期化
        Args:
            detection_results: AI推論結果（JSON形式）
            video_path: 入力動画のパス
            output_path: 出力動画のパス
        """
        self.detection_results = detection_results
        self.video_path = video_path
        self.output_path = output_path or f"/tmp/tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # 色設定（各IDに異なる色を割り当て）
        self.colors = [
            (0, 255, 0),    # 緑 - Person ID 1
            (0, 0, 255),    # 赤 - Person ID 2  
            (255, 0, 0),    # 青 - Person ID 3
            (255, 255, 0),  # シアン - Person ID 4
            (255, 0, 255),  # マゼンタ - Person ID 5
            (0, 255, 255),  # 黄色 - Person ID 6
        ]
    
    def get_detection_for_frame(self, frame_number):
        """指定フレームの検出結果を取得（簡略版）"""
        # APIの基本結果形式に対応: {'boxes': [[x1,y1,x2,y2]], 'labels': ['person'], 'confidence_scores': [0.95]}
        if not self.detection_results or 'boxes' not in self.detection_results:
            return []
        
        boxes = self.detection_results.get('boxes', [])
        labels = self.detection_results.get('labels', [])
        confidences = self.detection_results.get('confidence_scores', [])
        
        persons = []
        for i, box in enumerate(boxes):
            if i < len(labels) and labels[i] == 'person':
                confidence = confidences[i] if i < len(confidences) else 0.95
                persons.append({
                    'id': i + 1,
                    'bbox': box,
                    'confidence': confidence,
                    'label': 'person'
                })
        
        return persons
    
    def interpolate_detections(self, frame_number):
        """フレーム間の検出結果を補間（簡略版）"""
        # すべてのフレームで同じ検出結果を使用（シンプル版）
        return self.get_detection_for_frame(frame_number)
    
    def draw_person_roi(self, frame, person, frame_number, timestamp):
        """人物のROIを描画"""
        bbox = person.get('bbox', [0, 0, 0, 0])
        person_id = person.get('id', 0)
        confidence = person.get('confidence', 0.0)
        
        # バウンディングボックスの座標
        x1, y1, x2, y2 = map(int, bbox)
        
        # 色を取得（IDに基づく）
        color = self.colors[person_id % len(self.colors)]
        
        # バウンディングボックスを描画（太線）
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # ROI内部を半透明で塗りつぶし
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
        
        # ラベル背景
        label = f"ID:{person_id} ({confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # ラベル背景を描画
        cv2.rectangle(frame, (x1, y1-30), (x1 + label_size[0] + 10, y1), color, -1)
        
        # ラベルテキストを描画
        cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 中心点にクロスハイヤーを描画
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.drawMarker(frame, (center_x, center_y), color, cv2.MARKER_CROSS, 20, 3)
        
        return frame
    
    def draw_tracking_info(self, frame, frame_number, timestamp, total_persons):
        """追跡情報をフレームに描画"""
        height, width = frame.shape[:2]
        
        # 背景パネル
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 情報テキスト
        info_texts = [
            f"Frame: {frame_number}",
            f"Time: {timestamp}",
            f"Detected Persons: {total_persons}",
            f"Powered by Grounded-SAM-2"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 25 + (i * 20)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self):
        """動画を処理してROIトラッキング動画を生成"""
        print(f"🎬 動画処理開始: {self.detection_results}")
        
        # 入力動画パスを文字列として確保
        video_path_str = str(self.video_path)
        print(f"📁 動画ファイルパス: {video_path_str}")
        
        # 動画を開く
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise Exception(f"動画ファイルを開けません: {video_path_str}")
        
        # 動画情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム")
        
        # 動画ライターを設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_number = 1
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # タイムスタンプを計算
                timestamp_seconds = (frame_number - 1) / fps
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                milliseconds = int((timestamp_seconds % 1) * 1000)
                timestamp = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                
                # このフレームの検出結果を取得（補間含む）
                detected_persons = self.interpolate_detections(frame_number)
                
                # 各検出された人物にROIを描画
                for person in detected_persons:
                    frame = self.draw_person_roi(frame, person, frame_number, timestamp)
                
                # 追跡情報を描画
                frame = self.draw_tracking_info(frame, frame_number, timestamp, len(detected_persons))
                
                # フレームを出力
                out.write(frame)
                processed_frames += 1
                
                # 進捗表示
                if processed_frames % 30 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"🔄 処理進捗: {progress:.1f}% ({processed_frames}/{total_frames} フレーム)")
                
                frame_number += 1
                
        except Exception as e:
            print(f"❌ 動画処理エラー: {e}")
            raise
        finally:
            # リソースを解放
            cap.release()
            out.release()
        
        print(f"✅ 動画処理完了: {self.output_path}")
        print(f"📁 出力ファイル: {processed_frames} フレーム処理済み")
        
        return self.output_path
    
    def get_video_info(self):
        """処理済み動画の情報を取得"""
        if not os.path.exists(self.output_path):
            return None
            
        # ファイルサイズを取得
        file_size = os.path.getsize(self.output_path)
        
        return {
            'output_path': self.output_path,
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'created_at': datetime.now().isoformat()
        }

def create_tracking_video(detection_results, input_video_path):
    """ROIトラッキング動画を作成するメイン関数"""
    try:
        print("🎯 ROIトラッキング動画作成開始...")
        
        # トラッカーを初期化
        tracker = PersonTracker(detection_results, input_video_path)
        
        # 動画を処理
        output_path = tracker.process_video()
        
        # 結果情報を取得
        video_info = tracker.get_video_info()
        
        return {
            'success': True,
            'output_path': output_path,
            'video_info': video_info,
            'message': 'ROIトラッキング動画の作成が完了しました'
        }
        
    except Exception as e:
        print(f"❌ トラッキング動画作成エラー: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'ROIトラッキング動画の作成に失敗しました'
        }