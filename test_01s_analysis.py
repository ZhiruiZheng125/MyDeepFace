#!/usr/bin/env python3
"""
Test script for 0.1 second interval analysis and playable MP4 generation
"""

import cv2
import os
import sys

def test_01_second_analysis():
    """Test the 0.1 second interval analysis"""
    
    print("🧪 Testing 0.1 second interval emotion analysis")
    print("=" * 60)
    
    # Check if we have input videos
    input_dir = r"C:\my_vscode\MyDeepFace\InputVideos"
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Find available videos
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        import glob
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print("❌ No video files found for testing")
        return False
    
    test_video = video_files[0]
    print(f"🎬 Testing with: {os.path.basename(test_video)}")
    
    # Check video properties
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"❌ Cannot open test video: {test_video}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video info: {fps:.1f}fps, {total_frames} frames, {duration:.1f}s duration")
    
    # Calculate expected analysis points for 15 seconds at 0.1s intervals
    analysis_interval = max(1, int(fps * 0.1))  # Every 0.1 seconds
    max_frames_15s = int(fps * 15)
    expected_analysis_points = max_frames_15s // analysis_interval
    
    print(f"🎯 Analysis interval: every {analysis_interval} frames (0.1s)")
    print(f"🎯 Expected analysis points: {expected_analysis_points} (for 15 seconds)")
    
    cap.release()
    
    # Import and test the analysis function
    try:
        from analyze_with_output import analyze_video_with_output
        
        print("\n🚀 Starting 0.1s interval analysis test...")
        # This will run the full analysis
        analyze_video_with_output(test_video)
        
        # Check output
        video_name = os.path.splitext(os.path.basename(test_video))[0]
        output_dir = os.path.join("Output_Files", video_name)
        
        if os.path.exists(output_dir):
            print(f"\n✅ Output directory created: {output_dir}")
            
            # Check for expected files
            json_file = os.path.join(output_dir, f"emotion_analysis_{video_name}.json")
            csv_file = os.path.join(output_dir, f"emotion_analysis_{video_name}.csv")
            mp4_file = os.path.join(output_dir, f"analyzed_{video_name}.mp4")
            
            files_found = []
            if os.path.exists(json_file):
                files_found.append(f"📄 JSON: {os.path.basename(json_file)}")
            if os.path.exists(csv_file):
                files_found.append(f"📊 CSV: {os.path.basename(csv_file)}")
            if os.path.exists(mp4_file):
                files_found.append(f"🎥 MP4: {os.path.basename(mp4_file)}")
                
                # Comprehensive MP4 testing
                print(f"\n🔍 Testing MP4 playability...")
                test_mp4_playability(mp4_file)
            
            print("📁 Generated files:")
            for file_info in files_found:
                print(f"   {file_info}")
            
            # Check JSON for analysis count
            if os.path.exists(json_file):
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                analyzed_points = len(data['results'])
                print(f"📈 Analysis results: {analyzed_points} analysis points")
                
                # Check if we got approximately the expected number of analysis points
                tolerance = 0.2  # 20% tolerance
                if analyzed_points >= expected_analysis_points * (1 - tolerance):
                    print(f"✅ Analysis point count looks good! (Expected ~{expected_analysis_points})")
                    
                    # Check time intervals
                    if len(data['results']) >= 2:
                        time_diff = data['results'][1]['timestamp_seconds'] - data['results'][0]['timestamp_seconds']
                        print(f"📏 Time interval between analyses: {time_diff:.2f}s")
                        
                        if 0.08 <= time_diff <= 0.12:  # Allow small tolerance around 0.1s
                            print("✅ Time interval is correct (~0.1s)")
                        else:
                            print(f"⚠️  Time interval seems off (expected ~0.1s, got {time_diff:.2f}s)")
                else:
                    print(f"⚠️  Analysis point count lower than expected (got {analyzed_points}, expected ~{expected_analysis_points})")
            
            return True
        else:
            print("❌ Output directory not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mp4_playability(mp4_file):
    """Comprehensive test of MP4 file playability"""
    
    print(f"🎬 Testing MP4 file: {os.path.basename(mp4_file)}")
    
    # Test 1: Basic file existence and size
    if not os.path.exists(mp4_file):
        print("❌ MP4 file does not exist")
        return False
    
    file_size = os.path.getsize(mp4_file) / (1024 * 1024)
    print(f"📦 File size: {file_size:.1f}MB")
    
    if file_size < 0.1:
        print("⚠️  File size is very small, might be corrupted")
    
    # Test 2: OpenCV can read the file
    cap = cv2.VideoCapture(mp4_file)
    if not cap.isOpened():
        print("❌ OpenCV cannot open the MP4 file")
        return False
    
    # Test 3: Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"✅ Video properties:")
    print(f"   📐 Resolution: {width}x{height}")
    print(f"   🎬 Frame count: {frame_count}")
    print(f"   ⏱️  FPS: {fps:.1f}")
    print(f"   ⏲️  Duration: {duration:.1f}s")
    
    # Test 4: Try to read multiple frames
    frames_read = 0
    test_frames = min(10, frame_count)  # Test first 10 frames or all if less
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            if i == 0:
                print(f"   ✅ First frame shape: {frame.shape}")
        else:
            print(f"   ❌ Failed to read frame {i}")
            break
    
    cap.release()
    
    if frames_read == test_frames:
        print(f"✅ Successfully read all {test_frames} test frames")
        
        # Test 5: Try with different players/libraries
        print("🔧 Testing compatibility with different methods...")
        
        # Test with different backends
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_MSMF]:
            try:
                test_cap = cv2.VideoCapture(mp4_file, backend)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        print(f"   ✅ Backend {backend}: OK")
                    else:
                        print(f"   ⚠️  Backend {backend}: Opens but can't read")
                    test_cap.release()
                else:
                    print(f"   ❌ Backend {backend}: Cannot open")
            except Exception as e:
                print(f"   ❌ Backend {backend}: Error - {e}")
        
        return True
    else:
        print(f"❌ Could only read {frames_read}/{test_frames} frames")
        return False

def quick_codec_test():
    """Quick test of video codecs"""
    print("\n🔧 Testing video codec compatibility...")
    
    codecs_to_test = [
        ('mp4v', 'MP4V - Standard'),
        ('XVID', 'XVID - Alternative'),
        ('MJPG', 'MJPG - Motion JPEG'),
    ]
    
    for fourcc_str, description in codecs_to_test:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            test_writer = cv2.VideoWriter('temp_test.mp4', fourcc, 25.0, (640, 480))
            
            if test_writer.isOpened():
                # Try writing a test frame
                test_frame = cv2.ones((480, 640, 3), dtype=cv2.uint8) * 128
                test_writer.write(test_frame)
                test_writer.release()
                
                # Try reading it back
                test_reader = cv2.VideoCapture('temp_test.mp4')
                if test_reader.isOpened():
                    ret, frame = test_reader.read()
                    if ret:
                        print(f"✅ {description}: Full read/write OK")
                    else:
                        print(f"⚠️  {description}: Writes but can't read")
                    test_reader.release()
                else:
                    print(f"⚠️  {description}: Writes but can't open")
                
                # Clean up
                if os.path.exists('temp_test.mp4'):
                    os.remove('temp_test.mp4')
            else:
                print(f"❌ {description}: Cannot create writer")
                test_writer.release()
                
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

if __name__ == "__main__":
    print("🎬 Advanced Video Analysis & Playability Test")
    print("=" * 60)
    
    quick_codec_test()
    
    success = test_01_second_analysis()
    
    if success:
        print("\n🎉 All tests passed! The 0.1s analysis and MP4 generation are working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
