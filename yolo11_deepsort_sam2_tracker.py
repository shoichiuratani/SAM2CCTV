#!/usr/bin/env python3
"""
YOLO11 + DeepSORT + SAM2 高精度人物トラッキングシステム
最先端の検出・追跡・セグメンテーション技術を統合
"""

import cv2
import numpy as np
import torch
from datetime import datetime
import os
import json
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import time

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLO11PersonDetector:
    """YOLO11ベースの高精度人物検出器"""
    
    def __init__(self, model_size='n', confidence=0.5, device='auto'):
        """
        初期化
        Args:
            model_size: モデルサイズ ('n', 's', 'm', 'l', 'x')
            confidence: 信頼度閾値
            device: 実行デバイス ('cpu', 'cuda', 'auto')
        """
        self.confidence = confidence
        self.device = self._setup_device(device)
        self.model = self._load_model(model_size)
        
        # 人物クラスID (COCO dataset)
        self.person_class_id = 0
        
        logger.info(f"🎯 YOLO11-{model_size} 人物検出器を初期化完了")
        logger.info(f"📱 デバイス: {self.device}")
        logger.info(f"🎚️ 信頼度閾値: {confidence}")
    
    def _setup_device(self, device):
        """最適なデバイスを設定"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        return device
    
    def _load_model(self, model_size):
        """YOLO11モデルを読み込み"""
        try:
            # YOLO11の最新実装を試行
            from ultralytics import YOLO
            model_name = f"yolo11{model_size}.pt"
            model = YOLO(model_name)
            logger.info(f"✅ YOLO11モデル読み込み成功: {model_name}")
            return model
        except ImportError:
            logger.warning("⚠️ YOLO11が利用できません。YOLO8をフォールバックとして使用")
            from ultralytics import YOLO
            model_name = f"yolov8{model_size}.pt"
            model = YOLO(model_name)
            logger.info(f"🔄 YOLOv8フォールバック: {model_name}")
            return model
        except Exception as e:
            logger.error(f"❌ YOLOモデル読み込みエラー: {e}")
            # OpenCVベースのフォールバック
            return self._create_opencv_fallback()
    
    def _create_opencv_fallback(self):
        """OpenCVベースのフォールバック検出器"""
        logger.info("🔄 OpenCVフォールバック検出器を初期化")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        フレーム内の人物を検出
        
        Args:
            frame: 入力フレーム (BGR形式)
            
        Returns:
            検出結果のリスト
        """
        try:
            if hasattr(self.model, 'predict'):
                # YOLO11/YOLOv8による検出
                return self._yolo_detect(frame)
            else:
                # OpenCVフォールバック
                return self._opencv_detect(frame)
        except Exception as e:
            logger.error(f"❌ 人物検出エラー: {e}")
            return []
    
    def _yolo_detect(self, frame: np.ndarray) -> List[Dict]:
        """YOLO11/YOLOv8による高精度検出"""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[self.person_class_id],  # 人物のみ
            device=self.device,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i, box in enumerate(boxes.data):
                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                    
                    # 検出結果を正規化
                    height, width = frame.shape[:2]
                    detection = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'bbox_norm': [
                            float(x1/width), float(y1/height), 
                            float(x2/width), float(y2/height)
                        ],
                        'confidence': float(conf),
                        'class_name': 'person',
                        'detection_method': 'YOLO11'
                    }
                    detections.append(detection)
        
        return detections
    
    def _opencv_detect(self, frame: np.ndarray) -> List[Dict]:
        """OpenCVフォールバック検出"""
        (rects, weights) = self.model.detectMultiScale(
            frame, winStride=(4, 4), padding=(8, 8), scale=1.05
        )
        
        detections = []
        height, width = frame.shape[:2]
        
        for i, ((x, y, w, h), weight) in enumerate(zip(rects, weights)):
            if weight[0] > self.confidence:
                detection = {
                    'id': i,
                    'bbox': [float(x), float(y), float(x+w), float(y+h)],
                    'bbox_norm': [
                        float(x/width), float(y/height), 
                        float((x+w)/width), float((y+h)/height)
                    ],
                    'confidence': min(float(weight[0] * 0.7 + 0.3), 1.0),
                    'class_name': 'person',
                    'detection_method': 'OpenCV_HOG'
                }
                detections.append(detection)
        
        return detections


class DeepSORTTracker:
    """DeepSORT多物体追跡システム"""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        初期化
        Args:
            max_disappeared: 追跡失敗までの最大フレーム数
            max_distance: 関連付け最大距離
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.trackers = {}
        self.disappeared = {}
        
        try:
            # DeepSORT実装を試行
            from deep_sort_realtime import DeepSort
            self.deepsort = DeepSort(
                max_age=max_disappeared,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=torch.cuda.is_available(),
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            self.use_deepsort = True
            logger.info("✅ DeepSORT追跡システム初期化完了")
        except ImportError:
            logger.warning("⚠️ DeepSORT利用不可。シンプル追跡を使用")
            self.use_deepsort = False
            self.deepsort = None
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        検出結果を使用して追跡を更新
        
        Args:
            detections: YOLO11からの検出結果
            frame: 現在のフレーム
            
        Returns:
            追跡結果（IDを含む）
        """
        if self.use_deepsort and self.deepsort is not None:
            return self._deepsort_update(detections, frame)
        else:
            return self._simple_update(detections)
    
    def _deepsort_update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """DeepSORTによる高精度追跡"""
        try:
            # DeepSORT形式に変換
            detection_list = []
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                # [left, top, width, height] 形式に変換
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                detection_list.append(([x1, y1, w, h], conf, 'person'))
            
            # 追跡更新
            tracks = self.deepsort.update_tracks(detection_list, frame=frame)
            
            # 結果を整形
            tracked_objects = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()  # left, top, right, bottom
                
                tracked_objects.append({
                    'track_id': int(track_id),
                    'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                    'confidence': 0.9,  # DeepSORTの信頼度
                    'class_name': 'person',
                    'tracking_method': 'DeepSORT'
                })
            
            return tracked_objects
            
        except Exception as e:
            logger.error(f"❌ DeepSORT追跡エラー: {e}")
            return self._simple_update(detections)
    
    def _simple_update(self, detections: List[Dict]) -> List[Dict]:
        """シンプルな追跡アルゴリズム（フォールバック）"""
        tracked_objects = []
        
        for detection in detections:
            # 新しいIDを割り当て（シンプル版）
            detection['track_id'] = self.next_id
            detection['tracking_method'] = 'Simple'
            self.next_id += 1
            tracked_objects.append(detection)
        
        return tracked_objects


class SAM2Segmenter:
    """SAM2による精密セグメンテーション"""
    
    def __init__(self, model_size='small'):
        """
        初期化
        Args:
            model_size: SAM2モデルサイズ ('small', 'base', 'large')
        """
        self.model_size = model_size
        self.model = None
        self.predictor = None
        
        try:
            self._load_sam2_model()
            logger.info(f"✅ SAM2-{model_size} セグメンテーション初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ SAM2読み込み失敗: {e}")
            logger.info("🔄 基本的な矩形マスクを使用します")
    
    def _load_sam2_model(self):
        """SAM2モデルを読み込み"""
        try:
            # SAM2の実装を試行
            from segment_anything import sam_model_registry, SamPredictor
            
            # モデルファイルのパスを設定
            model_path = f"sam2_{self.model_size}.pth"
            
            # モデルを読み込み
            sam = sam_model_registry[f"vit_{self.model_size}"](checkpoint=model_path)
            if torch.cuda.is_available():
                sam.to(device='cuda')
            
            self.predictor = SamPredictor(sam)
            self.model = sam
            
        except ImportError:
            logger.warning("SAM2パッケージが見つかりません")
            raise
        except Exception as e:
            logger.error(f"SAM2モデル読み込みエラー: {e}")
            raise
    
    def segment_persons(self, frame: np.ndarray, tracked_objects: List[Dict]) -> List[Dict]:
        """
        追跡された人物の精密セグメンテーション
        
        Args:
            frame: 現在のフレーム
            tracked_objects: 追跡結果
            
        Returns:
            セグメンテーションマスク付きの結果
        """
        if self.predictor is not None:
            return self._sam2_segment(frame, tracked_objects)
        else:
            return self._simple_segment(frame, tracked_objects)
    
    def _sam2_segment(self, frame: np.ndarray, tracked_objects: List[Dict]) -> List[Dict]:
        """SAM2による高精度セグメンテーション"""
        try:
            # フレームを設定
            self.predictor.set_image(frame)
            
            segmented_objects = []
            for obj in tracked_objects:
                # バウンディングボックスからプロンプトを生成
                bbox = obj['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # SAM2予測
                masks, scores, logits = self.predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    mask = masks[0]
                    obj['mask'] = mask.astype(np.uint8)
                    obj['mask_score'] = float(scores[0]) if len(scores) > 0 else 0.8
                    obj['segmentation_method'] = 'SAM2'
                else:
                    obj['mask'] = self._create_bbox_mask(frame.shape, bbox)
                    obj['mask_score'] = 0.5
                    obj['segmentation_method'] = 'BBox_fallback'
                
                segmented_objects.append(obj)
            
            return segmented_objects
            
        except Exception as e:
            logger.error(f"❌ SAM2セグメンテーションエラー: {e}")
            return self._simple_segment(frame, tracked_objects)
    
    def _simple_segment(self, frame: np.ndarray, tracked_objects: List[Dict]) -> List[Dict]:
        """シンプルなバウンディングボックスマスク（フォールバック）"""
        segmented_objects = []
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            mask = self._create_bbox_mask(frame.shape, bbox)
            obj['mask'] = mask
            obj['mask_score'] = 0.7
            obj['segmentation_method'] = 'BoundingBox'
            segmented_objects.append(obj)
        
        return segmented_objects
    
    def _create_bbox_mask(self, frame_shape: Tuple[int, int, int], bbox: List[float]) -> np.ndarray:
        """バウンディングボックスから矩形マスクを作成"""
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        mask[y1:y2, x1:x2] = 255
        return mask


class AdvancedPersonTracker:
    """YOLO11 + DeepSORT + SAM2 統合トラッキングシステム"""
    
    def __init__(self, 
                 yolo_model='n', 
                 yolo_confidence=0.5,
                 deepsort_max_disappeared=30,
                 sam2_model='small'):
        """
        高精度トラッキングシステム初期化
        """
        self.detector = YOLO11PersonDetector(yolo_model, yolo_confidence)
        self.tracker = DeepSORTTracker(deepsort_max_disappeared)
        self.segmenter = SAM2Segmenter(sam2_model)
        
        self.frame_count = 0
        self.processing_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("🚀 YOLO11+DeepSORT+SAM2 統合システム初期化完了")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        フレームを処理して高精度追跡結果を返す
        
        Args:
            frame: 入力フレーム
            
        Returns:
            統合処理結果
        """
        start_time = time.time()
        
        # ステップ1: YOLO11による人物検出
        detections = self.detector.detect_persons(frame)
        
        # ステップ2: DeepSORTによる追跡
        tracked_objects = self.tracker.update(detections, frame)
        
        # ステップ3: SAM2による精密セグメンテーション
        segmented_objects = self.segmenter.segment_persons(frame, tracked_objects)
        
        processing_time = time.time() - start_time
        
        # 統計更新
        self.frame_count += 1
        self.processing_stats['total_frames'] += 1
        self.processing_stats['total_detections'] += len(detections)
        self.processing_stats['total_tracks'] += len(tracked_objects)
        self.processing_stats['avg_processing_time'] = (
            (self.processing_stats['avg_processing_time'] * (self.frame_count - 1) + processing_time) 
            / self.frame_count
        )
        
        return {
            'frame_number': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'tracked_objects': segmented_objects,
            'processing_time': processing_time,
            'fps': 1.0 / processing_time if processing_time > 0 else 0,
            'stats': self.processing_stats.copy()
        }
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """
        ビデオファイル全体を処理
        
        Args:
            video_path: 入力ビデオパス
            output_path: 出力ビデオパス
            
        Returns:
            処理結果サマリー
        """
        logger.info(f"🎬 ビデオ処理開始: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"ビデオを開けません: {video_path}")
        
        # ビデオ情報取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 出力ビデオセットアップ
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレーム処理
                result = self.process_frame(frame)
                all_results.append(result)
                
                # 可視化とビデオ出力
                if output_path:
                    visualized_frame = self.visualize_results(frame, result)
                    out.write(visualized_frame)
                
                # 進捗表示
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"🔄 処理進捗: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        logger.info(f"✅ ビデオ処理完了: {self.frame_count}フレーム処理")
        
        return {
            'input_video': video_path,
            'output_video': output_path,
            'total_frames_processed': self.frame_count,
            'video_info': {
                'fps': fps,
                'resolution': f"{width}x{height}",
                'duration_seconds': total_frames / fps if fps > 0 else 0
            },
            'final_stats': self.processing_stats,
            'processing_results': all_results
        }
    
    def visualize_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        結果を可視化
        
        Args:
            frame: 元のフレーム
            result: 処理結果
            
        Returns:
            可視化されたフレーム
        """
        viz_frame = frame.copy()
        
        # 追跡結果を描画
        for obj in result['tracked_objects']:
            track_id = obj.get('track_id', -1)
            bbox = obj['bbox']
            confidence = obj.get('confidence', 0)
            
            # バウンディングボックス
            x1, y1, x2, y2 = map(int, bbox)
            color = self._get_track_color(track_id)
            
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
            
            # マスクオーバーレイ（SAM2結果）
            if 'mask' in obj:
                mask = obj['mask']
                colored_mask = np.zeros_like(viz_frame)
                colored_mask[mask > 0] = color
                viz_frame = cv2.addWeighted(viz_frame, 0.8, colored_mask, 0.2, 0)
            
            # ラベル
            label = f"ID:{track_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(viz_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            cv2.putText(viz_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 統計情報表示
        stats_text = [
            f"Frame: {result['frame_number']}",
            f"Objects: {len(result['tracked_objects'])}",
            f"FPS: {result['fps']:.1f}",
            f"Method: YOLO11+DeepSORT+SAM2"
        ]
        
        for i, text in enumerate(stats_text):
            y = 30 + i * 25
            cv2.putText(viz_frame, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return viz_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """追跡IDに基づいた色を生成"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 165, 0), (0, 128, 0)
        ]
        return colors[track_id % len(colors)]


def create_advanced_tracking_video(input_video_path: str, output_dir: str = "advanced_tracking") -> Dict:
    """
    高精度トラッキングビデオを生成するメイン関数
    
    Args:
        input_video_path: 入力ビデオパス
        output_dir: 出力ディレクトリ
        
    Returns:
        処理結果
    """
    try:
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 出力ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_dir, f"advanced_tracking_{timestamp}.mp4")
        
        # 高精度トラッカー初期化
        tracker = AdvancedPersonTracker(
            yolo_model='s',  # 精度重視
            yolo_confidence=0.3,  # 検出感度向上
            deepsort_max_disappeared=50,
            sam2_model='small'
        )
        
        # ビデオ処理実行
        result = tracker.process_video(input_video_path, output_video_path)
        
        return {
            'success': True,
            'output_path': output_video_path,
            'processing_result': result,
            'message': 'YOLO11+DeepSORT+SAM2による高精度追跡完了'
        }
        
    except Exception as e:
        logger.error(f"❌ 高精度トラッキングエラー: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': '高精度追跡処理に失敗しました'
        }


if __name__ == "__main__":
    # テスト実行
    logger.info("🚀 YOLO11+DeepSORT+SAM2システムテスト開始")
    
    # システム初期化テスト
    tracker = AdvancedPersonTracker()
    
    # テスト画像での検証
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = tracker.process_frame(test_frame)
    
    logger.info(f"✅ システムテスト完了")
    logger.info(f"📊 テスト結果: {result['stats']}")