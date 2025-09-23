#!/usr/bin/env python3
"""
包括的な人物トラッキングテストシステム
YOLO11+DeepSORT+SAM2の全機能テスト
"""

import cv2
import numpy as np
import os
import time
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import argparse
from yolo11_deepsort_sam2_tracker import AdvancedPersonTracker, create_advanced_tracking_video

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTrackingTester:
    """包括的トラッキングテスター"""
    
    def __init__(self):
        self.results = []
        self.tracker = None
        
    def initialize_tracker(self, model_size='n', confidence=0.3):
        """高精度トラッカーを初期化"""
        logger.info(f"🚀 YOLO11-{model_size}トラッカー初期化中...")
        
        self.tracker = AdvancedPersonTracker(
            yolo_model=model_size,
            yolo_confidence=confidence,
            deepsort_max_disappeared=50,
            sam2_model='small'
        )
        
        logger.info("✅ 高精度トラッカー初期化完了")
        
    def test_single_frame(self, image_path: str) -> Dict:
        """単一画像での検出テスト"""
        logger.info(f"🖼️ 単一フレームテスト: {image_path}")
        
        if not self.tracker:
            raise ValueError("トラッカーが初期化されていません")
        
        # 画像読み込み
        frame = cv2.imread(image_path)
        if frame is None:
            return {'success': False, 'error': f'画像読み込み失敗: {image_path}'}
        
        start_time = time.time()
        result = self.tracker.process_frame(frame)
        processing_time = time.time() - start_time
        
        # JSONシリアライゼーション対応
        serializable_detections = []
        for det in result['detections']:
            serializable_det = {}
            for k, v in det.items():
                if isinstance(v, np.ndarray):
                    serializable_det[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    serializable_det[k] = float(v)
                else:
                    serializable_det[k] = v
            serializable_detections.append(serializable_det)
        
        test_result = {
            'test_type': 'single_frame',
            'input_file': image_path,
            'success': True,
            'processing_time': processing_time,
            'detections': len(result['detections']),
            'tracked_objects': len(result['tracked_objects']),
            'detection_details': serializable_detections,
            'frame_shape': list(frame.shape)  # numpy配列をリストに変換
        }
        
        self.results.append(test_result)
        logger.info(f"✅ 検出数: {test_result['detections']}, 処理時間: {processing_time:.3f}s")
        
        return test_result
    
    def test_video_file(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """動画ファイルでのテスト"""
        logger.info(f"🎬 動画ファイルテスト: {video_path}")
        
        if not self.tracker:
            raise ValueError("トラッカーが初期化されていません")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': f'動画読み込み失敗: {video_path}'}
        
        # 動画情報取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"📊 動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム")
        
        frame_results = []
        total_detections = 0
        total_processing_time = 0
        
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレーム処理
            frame_start = time.time()
            result = self.tracker.process_frame(frame)
            frame_time = time.time() - frame_start
            
            detections = len(result['detections'])
            tracked_objects = len(result['tracked_objects'])
            
            total_detections += detections
            total_processing_time += frame_time
            
            # JSONシリアライゼーション対応の検出詳細
            serializable_detections = []
            for det in result['detections'][:3]:  # 最初の3件のみ保存
                serializable_det = {}
                for k, v in det.items():
                    if isinstance(v, np.ndarray):
                        serializable_det[k] = v.tolist()  # numpy配列をリストに変換
                    elif isinstance(v, (np.integer, np.floating)):
                        serializable_det[k] = float(v)  # numpy数値型を標準型に変換
                    else:
                        serializable_det[k] = v
                serializable_detections.append(serializable_det)
            
            frame_result = {
                'frame_idx': frame_idx,
                'detections': detections,
                'tracked_objects': tracked_objects,
                'processing_time': frame_time,
                'detection_details': serializable_detections
            }
            frame_results.append(frame_result)
            
            # 進捗表示
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                progress = (frame_idx + 1) / total_frames * 100
                logger.info(f"進捗 {progress:.1f}% ({frame_idx+1}/{total_frames}) - 検出: {detections}")
        
        cap.release()
        total_time = time.time() - start_time
        
        test_result = {
            'test_type': 'video_file',
            'input_file': video_path,
            'success': True,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            },
            'processing_stats': {
                'total_time': total_time,
                'total_processing_time': total_processing_time,
                'avg_fps': total_frames / total_processing_time if total_processing_time > 0 else 0,
                'real_time_factor': total_processing_time / (total_frames / fps) if fps > 0 else 0
            },
            'detection_stats': {
                'total_detections': total_detections,
                'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
                'unique_tracks': len(set(
                    obj.get('track_id')
                    for frame in frame_results
                    for obj in frame['detection_details']
                    if obj.get('track_id') is not None
                ))
            },
            'frame_results': frame_results
        }
        
        self.results.append(test_result)
        
        logger.info(f"✅ 動画テスト完了")
        logger.info(f"📊 統計: 総検出数={total_detections}, 平均FPS={test_result['processing_stats']['avg_fps']:.2f}")
        
        return test_result
    
    def test_webcam(self, duration_seconds: int = 10, camera_id: int = 0) -> Dict:
        """Webカメラでのリアルタイムテスト"""
        logger.info(f"📹 Webカメラテスト開始: {duration_seconds}秒間, カメラID={camera_id}")
        
        if not self.tracker:
            raise ValueError("トラッカーが初期化されていません")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {'success': False, 'error': f'Webカメラ接続失敗: ID={camera_id}'}
        
        # カメラ設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_results = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # フレーム処理
            frame_start = time.time()
            result = self.tracker.process_frame(frame)
            frame_time = time.time() - frame_start
            
            frame_result = {
                'timestamp': time.time() - start_time,
                'detections': len(result['detections']),
                'tracked_objects': len(result['tracked_objects']),
                'processing_time': frame_time
            }
            frame_results.append(frame_result)
            
            # リアルタイム表示（オプション）
            elapsed = time.time() - start_time
            logger.info(f"⏱️ {elapsed:.1f}s - 検出: {frame_result['detections']}, FPS: {1/frame_time:.1f}")
        
        cap.release()
        
        total_frames = len(frame_results)
        total_detections = sum(r['detections'] for r in frame_results)
        avg_processing_time = sum(r['processing_time'] for r in frame_results) / total_frames if total_frames > 0 else 0
        
        test_result = {
            'test_type': 'webcam',
            'success': True,
            'duration': duration_seconds,
            'camera_id': camera_id,
            'processing_stats': {
                'total_frames': total_frames,
                'avg_fps': 1 / avg_processing_time if avg_processing_time > 0 else 0,
                'avg_processing_time': avg_processing_time
            },
            'detection_stats': {
                'total_detections': total_detections,
                'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
            },
            'frame_results': frame_results
        }
        
        self.results.append(test_result)
        
        logger.info(f"✅ Webカメラテスト完了")
        logger.info(f"📊 平均FPS: {test_result['processing_stats']['avg_fps']:.2f}")
        
        return test_result
    
    def benchmark_performance(self, test_video_path: str) -> Dict:
        """パフォーマンスベンチマーク"""
        logger.info("🏃‍♂️ パフォーマンスベンチマーク実行中...")
        
        results = {}
        
        # 異なるモデルサイズでのテスト
        model_sizes = ['n', 's']  # nano, small
        
        for model_size in model_sizes:
            logger.info(f"🔧 モデル {model_size} でテスト中...")
            
            # トラッカー初期化
            self.initialize_tracker(model_size, confidence=0.3)
            
            # ベンチマークテスト
            test_result = self.test_video_file(test_video_path, max_frames=30)
            
            if test_result['success']:
                results[f'yolo11{model_size}'] = {
                    'model_size': model_size,
                    'avg_fps': test_result['processing_stats']['avg_fps'],
                    'avg_detections': test_result['detection_stats']['avg_detections_per_frame'],
                    'total_detections': test_result['detection_stats']['total_detections'],
                    'real_time_factor': test_result['processing_stats']['real_time_factor']
                }
        
        benchmark_result = {
            'test_type': 'benchmark',
            'success': True,
            'model_results': results,
            'best_model': max(results.keys(), key=lambda k: results[k]['avg_fps']) if results else None
        }
        
        self.results.append(benchmark_result)
        
        logger.info("✅ ベンチマーク完了")
        for model, stats in results.items():
            logger.info(f"📊 {model}: FPS={stats['avg_fps']:.2f}, 検出={stats['avg_detections']:.2f}")
        
        return benchmark_result
    
    def save_results(self, output_path: str = "tracking_test_results.json"):
        """テスト結果保存"""
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(self.results),
            'results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 テスト結果保存完了: {output_path}")
    
    def print_summary(self):
        """結果サマリー表示"""
        logger.info("\n" + "="*60)
        logger.info("📋 YOLO11+DeepSORT+SAM2 テスト結果サマリー")
        logger.info("="*60)
        
        for i, result in enumerate(self.results, 1):
            logger.info(f"\n🧪 テスト {i}: {result['test_type']}")
            
            if result.get('success'):
                if result['test_type'] == 'single_frame':
                    logger.info(f"   検出数: {result['detections']}")
                    logger.info(f"   処理時間: {result['processing_time']:.3f}s")
                    
                elif result['test_type'] == 'video_file':
                    stats = result['processing_stats']
                    detection_stats = result['detection_stats']
                    logger.info(f"   平均FPS: {stats['avg_fps']:.2f}")
                    logger.info(f"   総検出数: {detection_stats['total_detections']}")
                    logger.info(f"   ユニーク追跡数: {detection_stats['unique_tracks']}")
                    
                elif result['test_type'] == 'webcam':
                    stats = result['processing_stats']
                    detection_stats = result['detection_stats']
                    logger.info(f"   平均FPS: {stats['avg_fps']:.2f}")
                    logger.info(f"   総検出数: {detection_stats['total_detections']}")
                    
                elif result['test_type'] == 'benchmark':
                    logger.info(f"   最適モデル: {result['best_model']}")
                    for model, stats in result['model_results'].items():
                        logger.info(f"   {model}: FPS={stats['avg_fps']:.2f}")
            else:
                logger.error(f"   エラー: {result.get('error')}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='YOLO11+DeepSORT+SAM2 包括テスト')
    parser.add_argument('--test-type', choices=['frame', 'video', 'webcam', 'benchmark', 'all'], 
                        default='video', help='テストタイプ')
    parser.add_argument('--input', type=str, help='入力ファイルパス')
    parser.add_argument('--duration', type=int, default=10, help='Webカメラテスト時間(秒)')
    parser.add_argument('--camera-id', type=int, default=0, help='カメラID')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', help='YOLOモデルサイズ')
    parser.add_argument('--confidence', type=float, default=0.3, help='信頼度閾値')
    parser.add_argument('--output', type=str, default='tracking_test_results.json', help='結果保存ファイル')
    
    args = parser.parse_args()
    
    # テスター初期化
    tester = ComprehensiveTrackingTester()
    tester.initialize_tracker(args.model_size, args.confidence)
    
    try:
        if args.test_type == 'frame' or args.test_type == 'all':
            if args.input and os.path.exists(args.input):
                tester.test_single_frame(args.input)
            else:
                logger.warning("画像ファイルが指定されていないか存在しません")
        
        if args.test_type == 'video' or args.test_type == 'all':
            # デフォルトのテスト動画を使用
            test_videos = [
                args.input if args.input and os.path.exists(args.input) else None,
                'test_sample.mp4' if os.path.exists('test_sample.mp4') else None,
                'realistic_test.mp4' if os.path.exists('realistic_test.mp4') else None
            ]
            
            test_video = next((v for v in test_videos if v), None)
            
            if test_video:
                tester.test_video_file(test_video)
            else:
                logger.warning("テスト動画が見つかりません")
        
        if args.test_type == 'webcam' or args.test_type == 'all':
            tester.test_webcam(args.duration, args.camera_id)
        
        if args.test_type == 'benchmark' or args.test_type == 'all':
            test_video = args.input if args.input and os.path.exists(args.input) else 'realistic_test.mp4'
            if os.path.exists(test_video):
                tester.benchmark_performance(test_video)
            else:
                logger.warning("ベンチマーク用動画が見つかりません")
        
        # 結果表示・保存
        tester.print_summary()
        tester.save_results(args.output)
        
    except Exception as e:
        logger.error(f"❌ テスト実行エラー: {e}")
        raise

if __name__ == '__main__':
    main()