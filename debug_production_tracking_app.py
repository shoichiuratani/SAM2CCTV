#!/usr/bin/env python3
"""
デバッグ強化版 YOLO11+DeepSORT+SAM2 人物トラッキングWebアプリケーション
Background processing worker付きの修正版
"""

import os
import sys
import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import logging
import requests
from threading import Thread, Lock
import queue
import cv2
import numpy as np
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import tracking modules
try:
    from yolo11_deepsort_sam2_tracker import AdvancedPersonTracker, create_advanced_tracking_video
    ADVANCED_TRACKING_AVAILABLE = True
    logger.info("✅ YOLO11+DeepSORT+SAM2 system available")
except ImportError as e:
    ADVANCED_TRACKING_AVAILABLE = False
    logger.warning(f"⚠️ Advanced tracking unavailable: {e}")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='tracking_uploads_')
app.config['RESULTS_FOLDER'] = tempfile.mkdtemp(prefix='tracking_results_')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global state with thread locks
active_sessions = {}
sessions_lock = Lock()
processing_queue = queue.Queue()
system_stats = {
    'total_processed': 0,
    'successful_tracks': 0,
    'failed_processes': 0,
    'avg_processing_time': 0.0,
    'system_uptime': time.time()
}
stats_lock = Lock()

# Configuration
ADVANCED_TRACKING_SERVER_URL = "https://5002-isr1fqyzouakmq5p9u4zf-6532622b.e2b.dev"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'jpg', 'jpeg', 'png', 'gif'}

# Background processing worker flag
worker_running = False
worker_thread = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    """Check if file is a video"""
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def background_processing_worker():
    """Background worker to process jobs from queue"""
    global worker_running
    logger.info("🔄 Background processing worker started")
    
    while worker_running:
        try:
            # Get job from queue with timeout
            job_data = processing_queue.get(timeout=1)
            logger.info(f"🔄 Worker picked up job: {job_data['job_id']}")
            
            # Process the job
            result = process_tracking_job(job_data)
            logger.info(f"✅ Worker completed job: {job_data['job_id']} - Success: {result['success']}")
            
            # Mark task as done
            processing_queue.task_done()
            
        except queue.Empty:
            # No jobs in queue, continue waiting
            continue
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
            logger.error(f"❌ Worker traceback: {traceback.format_exc()}")
            processing_queue.task_done()
    
    logger.info("🔄 Background processing worker stopped")

def start_background_worker():
    """Start the background processing worker"""
    global worker_running, worker_thread
    
    if not worker_running:
        worker_running = True
        worker_thread = Thread(target=background_processing_worker, daemon=True)
        worker_thread.start()
        logger.info("🚀 Background worker thread started")
    else:
        logger.warning("⚠️ Background worker already running")

def stop_background_worker():
    """Stop the background processing worker"""
    global worker_running, worker_thread
    
    if worker_running:
        worker_running = False
        if worker_thread:
            worker_thread.join(timeout=5)
        logger.info("🛑 Background worker stopped")

@app.route('/')
def index():
    """Main application page"""
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    
    logger.debug(f"📊 Index page accessed by session: {session_id}")
    
    return render_template('production_index.html', 
                         session_id=session_id,
                         advanced_available=ADVANCED_TRACKING_AVAILABLE,
                         system_stats=get_system_stats())

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process tracking"""
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    
    logger.info(f"📤 Upload request from session: {session_id}")
    
    if 'file' not in request.files:
        logger.error("❌ No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("❌ No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.error(f"❌ File type not allowed: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{session_id[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        logger.info(f"💾 File saved: {file_path} (Size: {os.path.getsize(file_path)} bytes)")
        
        # Get processing parameters
        processing_mode = request.form.get('processing_mode', 'standard')
        confidence = float(request.form.get('confidence', 0.3))
        model_size = request.form.get('model_size', 'n')
        
        logger.info(f"🎯 Processing parameters: mode={processing_mode}, confidence={confidence}, model={model_size}")
        
        # Create processing job
        job_data = {
            'session_id': session_id,
            'job_id': str(uuid.uuid4()),
            'file_path': file_path,
            'original_filename': filename,
            'processing_mode': processing_mode,
            'confidence': confidence,
            'model_size': model_size,
            'is_video': is_video_file(filename),
            'status': 'queued',
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"📋 Created job: {job_data['job_id']} for file: {filename}")
        
        # Store job in active sessions
        with sessions_lock:
            active_sessions[job_data['job_id']] = job_data
        
        # Process based on mode
        if processing_mode == 'instant':
            logger.info(f"⚡ Processing instant job: {job_data['job_id']}")
            result = process_tracking_job(job_data)
            return jsonify(result)
        else:
            # Add to processing queue
            logger.info(f"📥 Adding job to queue: {job_data['job_id']}")
            processing_queue.put(job_data)
            logger.info(f"📊 Queue size: {processing_queue.qsize()}")
            
            return jsonify({
                'success': True,
                'job_id': job_data['job_id'],
                'status': 'queued',
                'message': 'File uploaded successfully, processing queued',
                'queue_size': processing_queue.qsize()
            })
    
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        logger.error(f"❌ Upload traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def process_tracking_job(job_data):
    """Process a tracking job with detailed logging"""
    job_id = job_data['job_id']
    
    try:
        logger.info(f"🔄 Starting processing for job {job_id}")
        logger.debug(f"📋 Job data: {job_data}")
        
        # Update job status to processing
        with sessions_lock:
            if job_id in active_sessions:
                active_sessions[job_id]['status'] = 'processing'
                active_sessions[job_id]['started_at'] = datetime.now().isoformat()
                logger.info(f"📝 Updated job status to processing: {job_id}")
        
        start_time = time.time()
        
        # Route to appropriate processing method
        if job_data['processing_mode'] == 'advanced_server':
            logger.info(f"🌐 Processing with remote advanced server: {job_id}")
            result = process_with_advanced_server(job_data)
        elif job_data['processing_mode'] == 'local_advanced':
            logger.info(f"🏠 Processing with local advanced tracking: {job_id}")
            result = process_with_local_advanced(job_data)
        else:
            logger.info(f"📱 Processing with standard tracking: {job_id}")
            result = process_with_standard_tracking(job_data)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Processing completed in {processing_time:.2f}s for job {job_id}")
        
        # Update job status
        with sessions_lock:
            if job_id in active_sessions:
                active_sessions[job_id]['status'] = 'completed' if result['success'] else 'failed'
                active_sessions[job_id]['completed_at'] = datetime.now().isoformat()
                active_sessions[job_id]['processing_time'] = processing_time
                active_sessions[job_id]['result'] = result
                logger.info(f"📝 Final job status: {active_sessions[job_id]['status']} for {job_id}")
        
        # Update system stats
        update_system_stats(processing_time, result['success'])
        
        logger.info(f"✅ Job {job_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed with error: {e}")
        logger.error(f"❌ Job {job_id} traceback: {traceback.format_exc()}")
        
        # Update job status to failed
        with sessions_lock:
            if job_id in active_sessions:
                active_sessions[job_id]['status'] = 'failed'
                active_sessions[job_id]['error'] = str(e)
                active_sessions[job_id]['completed_at'] = datetime.now().isoformat()
        
        update_system_stats(0, False)
        
        return {'success': False, 'error': str(e)}

def process_with_local_advanced(job_data):
    """Process using local advanced tracking with detailed logging"""
    try:
        if not ADVANCED_TRACKING_AVAILABLE:
            logger.error("❌ Local advanced tracking not available")
            return {'success': False, 'error': 'Local advanced tracking not available'}
        
        logger.info("🏠 Initializing local advanced tracker")
        
        # Initialize tracker with error handling
        try:
            tracker = AdvancedPersonTracker(
                yolo_model=job_data['model_size'],
                yolo_confidence=job_data['confidence'],
                deepsort_max_disappeared=30,
                sam2_model='small'
            )
            logger.info("✅ Tracker initialized successfully")
        except Exception as e:
            logger.error(f"❌ Tracker initialization failed: {e}")
            logger.error(f"❌ Tracker init traceback: {traceback.format_exc()}")
            return {'success': False, 'error': f'Tracker initialization failed: {e}'}
        
        if job_data['is_video']:
            logger.info(f"🎬 Processing video file: {job_data['file_path']}")
            
            # Verify file exists
            if not os.path.exists(job_data['file_path']):
                logger.error(f"❌ Video file not found: {job_data['file_path']}")
                return {'success': False, 'error': 'Video file not found'}
            
            # Get video info
            cap = cv2.VideoCapture(job_data['file_path'])
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {job_data['file_path']}")
                return {'success': False, 'error': 'Cannot open video file'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            logger.info(f"📊 Video info: {width}x{height}, {fps}fps, {frame_count} frames")
            
            # Process video with timeout for long videos
            max_frames = min(frame_count, 30)  # Limit to 30 frames for safety
            logger.info(f"🎯 Processing max {max_frames} frames")
            
            # Simple frame-by-frame processing for debugging
            cap = cv2.VideoCapture(job_data['file_path'])
            total_detections = 0
            processed_frames = 0
            
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    result = tracker.process_frame(frame)
                    detections = len(result['detections'])
                    total_detections += detections
                    processed_frames += 1
                    
                    if i % 5 == 0:  # Log every 5th frame
                        logger.info(f"📹 Frame {i+1}/{max_frames}: {detections} detections")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Frame {i} processing failed: {e}")
                    continue
            
            cap.release()
            
            logger.info(f"🎬 Video processing complete: {processed_frames} frames, {total_detections} total detections")
            
            return {
                'success': True,
                'processing_method': 'Local YOLO11+DeepSORT+SAM2 (Debug)',
                'stats': {
                    'frames_processed': processed_frames,
                    'total_detections': total_detections,
                    'avg_detections_per_frame': total_detections / processed_frames if processed_frames > 0 else 0,
                    'video_info': {
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'total_frames': frame_count
                    }
                },
                'technologies': ['YOLO11', 'DeepSORT', 'SAM2']
            }
        else:
            logger.info(f"🖼️ Processing image file: {job_data['file_path']}")
            
            # Verify file exists
            if not os.path.exists(job_data['file_path']):
                logger.error(f"❌ Image file not found: {job_data['file_path']}")
                return {'success': False, 'error': 'Image file not found'}
            
            # Process single image
            frame = cv2.imread(job_data['file_path'])
            if frame is None:
                logger.error(f"❌ Cannot read image: {job_data['file_path']}")
                return {'success': False, 'error': 'Cannot read image file'}
            
            logger.info(f"📊 Image shape: {frame.shape}")
            
            try:
                result = tracker.process_frame(frame)
                detections = len(result['detections'])
                
                logger.info(f"🎯 Image processing complete: {detections} detections")
                
                return {
                    'success': True,
                    'processing_method': 'Local YOLO11+DeepSORT+SAM2 (Debug)',
                    'stats': {
                        'detections': detections,
                        'tracked_objects': len(result['tracked_objects']),
                        'image_shape': frame.shape
                    },
                    'detection_details': result['detections'][:5],  # First 5 detections
                    'technologies': ['YOLO11', 'DeepSORT', 'SAM2']
                }
                
            except Exception as e:
                logger.error(f"❌ Image processing failed: {e}")
                logger.error(f"❌ Image processing traceback: {traceback.format_exc()}")
                return {'success': False, 'error': f'Image processing failed: {e}'}
            
    except Exception as e:
        logger.error(f"❌ Local advanced processing failed: {e}")
        logger.error(f"❌ Local advanced traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Local advanced processing failed: {e}'}

def process_with_advanced_server(job_data):
    """Process using remote advanced tracking server"""
    try:
        logger.info("🌐 Using remote advanced tracking server")
        
        with open(job_data['file_path'], 'rb') as f:
            files = {'file': (job_data['original_filename'], f, 'video/mp4' if job_data['is_video'] else 'image/jpeg')}
            data = {
                'confidence': job_data['confidence'],
                'output_format': 'json'
            }
            
            logger.info(f"📤 Sending request to server: {ADVANCED_TRACKING_SERVER_URL}")
            response = requests.post(
                f"{ADVANCED_TRACKING_SERVER_URL}/advanced_track",
                files=files,
                data=data,
                timeout=300  # 5 minutes
            )
        
        logger.info(f"📥 Server response: {response.status_code}")
        
        if response.status_code == 200:
            server_result = response.json()
            logger.info("✅ Remote server processing successful")
            
            return {
                'success': True,
                'processing_method': 'Remote YOLO11+DeepSORT+SAM2',
                'server_response': server_result,
                'stats': server_result.get('stats', {}),
                'technologies': ['YOLO11', 'DeepSORT', 'SAM2', 'Remote API']
            }
        else:
            logger.error(f"❌ Server error: {response.status_code}")
            return {
                'success': False,
                'error': f'Remote server error: {response.status_code}',
                'response_text': response.text[:500]
            }
            
    except Exception as e:
        logger.error(f"❌ Remote server processing failed: {e}")
        return {'success': False, 'error': f'Remote server processing failed: {e}'}

def process_with_standard_tracking(job_data):
    """Process using standard OpenCV tracking"""
    try:
        logger.info("📱 Using standard OpenCV tracking")
        
        if job_data['is_video']:
            # Basic video processing
            cap = cv2.VideoCapture(job_data['file_path'])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Process first few frames for demo
            detections_count = 0
            processed_frames = 0
            
            while cap.read()[0] and processed_frames < min(10, frame_count):
                processed_frames += 1
                # Simulate detection
                if processed_frames % 3 == 0:
                    detections_count += np.random.randint(0, 2)
            
            cap.release()
            
            logger.info(f"📹 Standard processing: {processed_frames} frames, {detections_count} simulated detections")
            
            return {
                'success': True,
                'processing_method': 'Standard OpenCV',
                'stats': {
                    'frames_processed': processed_frames,
                    'simulated_detections': detections_count,
                    'video_fps': fps
                },
                'technologies': ['OpenCV', 'Standard Tracking']
            }
        else:
            # Basic image processing
            frame = cv2.imread(job_data['file_path'])
            
            logger.info(f"🖼️ Standard image processing: shape={frame.shape if frame is not None else 'None'}")
            
            return {
                'success': True,
                'processing_method': 'Standard OpenCV',
                'stats': {
                    'image_processed': True,
                    'image_shape': frame.shape if frame is not None else None
                },
                'technologies': ['OpenCV', 'Standard Processing']
            }
            
    except Exception as e:
        logger.error(f"❌ Standard processing failed: {e}")
        return {'success': False, 'error': f'Standard processing failed: {e}'}

def update_system_stats(processing_time, success):
    """Update system statistics"""
    global system_stats
    
    with stats_lock:
        system_stats['total_processed'] += 1
        
        if success:
            system_stats['successful_tracks'] += 1
        else:
            system_stats['failed_processes'] += 1
        
        # Update average processing time
        current_avg = system_stats['avg_processing_time']
        total = system_stats['total_processed']
        system_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )

def get_system_stats():
    """Get current system statistics"""
    with stats_lock:
        uptime = time.time() - system_stats['system_uptime']
        
        return {
            **system_stats.copy(),
            'uptime_seconds': uptime,
            'uptime_formatted': f"{uptime/3600:.1f} hours",
            'success_rate': (
                system_stats['successful_tracks'] / system_stats['total_processed'] * 100
                if system_stats['total_processed'] > 0 else 0
            ),
            'worker_running': worker_running,
            'queue_size': processing_queue.qsize()
        }

@app.route('/api/job_status/<job_id>')
def get_job_status(job_id):
    """Get job processing status"""
    logger.debug(f"📊 Status check for job: {job_id}")
    
    with sessions_lock:
        if job_id not in active_sessions:
            logger.warning(f"⚠️ Job not found: {job_id}")
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = active_sessions[job_id].copy()
    
    logger.debug(f"📊 Job status: {job_data['status']} for {job_id}")
    
    return jsonify({
        'job_id': job_id,
        'status': job_data['status'],
        'created_at': job_data.get('created_at'),
        'started_at': job_data.get('started_at'),
        'completed_at': job_data.get('completed_at'),
        'processing_time': job_data.get('processing_time'),
        'result': job_data.get('result'),
        'error': job_data.get('error')
    })

@app.route('/api/download/<job_id>')
def download_result(job_id):
    """Download processed result file"""
    with sessions_lock:
        if job_id not in active_sessions:
            return jsonify({'error': 'Job not found'}), 404
        
        job_data = active_sessions[job_id]
    
    if job_data.get('status') != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    result = job_data.get('result', {})
    output_path = result.get('output_path')
    
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output file not available'}), 404
    
    return send_file(output_path, as_attachment=True)

@app.route('/api/system_stats')
def api_system_stats():
    """API endpoint for system statistics"""
    return jsonify(get_system_stats())

@app.route('/api/active_jobs')
def get_active_jobs():
    """Get list of active jobs for current session"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'jobs': []})
    
    with sessions_lock:
        session_jobs = [
            {
                'job_id': job_id,
                'original_filename': job_data['original_filename'],
                'status': job_data['status'],
                'processing_mode': job_data['processing_mode'],
                'created_at': job_data['created_at']
            }
            for job_id, job_data in active_sessions.items()
            if job_data['session_id'] == session_id
        ]
    
    return jsonify({'jobs': session_jobs})

# Debug endpoints
@app.route('/api/debug/queue_info')
def debug_queue_info():
    """Debug endpoint for queue information"""
    return jsonify({
        'queue_size': processing_queue.qsize(),
        'worker_running': worker_running,
        'active_sessions_count': len(active_sessions),
        'system_stats': get_system_stats()
    })

@app.route('/api/debug/active_sessions')
def debug_active_sessions():
    """Debug endpoint for all active sessions"""
    with sessions_lock:
        return jsonify({
            'active_sessions': active_sessions.copy(),
            'total_sessions': len(active_sessions)
        })

if __name__ == '__main__':
    logger.info("🚀 Starting Debug YOLO11+DeepSORT+SAM2 Tracking Application")
    logger.info(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
    logger.info(f"🔧 Advanced tracking available: {ADVANCED_TRACKING_AVAILABLE}")
    
    # Start background worker
    start_background_worker()
    
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5004)),
            debug=False
        )
    finally:
        # Cleanup
        stop_background_worker()
        logger.info("🛑 Application shutdown complete")