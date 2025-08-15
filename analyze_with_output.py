from deepface import DeepFace
import cv2
import os
import json
import pandas as pd
from datetime import datetime
import glob

def list_available_videos():
    """List all available video files and let user choose"""
    
    video_dir = r"C:\my_vscode\MyDeepFace\InputVideos"
    
    # Supported video formats
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        print("❌ No video files found in InputVideos directory")
        return None
    
    print("\n📹 Found the following video files:")
    print("=" * 60)
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Get video information
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"{i}. {filename}")
            print(f"   📐 Resolution: {width}x{height}")
            print(f"   ⏱️  Duration: {duration:.1f}s")
            print(f"   📦 Size: {size_mb:.1f}MB")
            print()
        else:
            print(f"{i}. {filename} (Unable to read video info)")
            print()
    
    # User selection
    while True:
        try:
            choice = input(f"Please select video to analyze (1-{len(video_files)}) or enter 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_video = video_files[choice_num - 1]
                print(f"\n✅ You selected: {os.path.basename(selected_video)}")
                return selected_video
            else:
                print(f"❌ Please enter a number between 1-{len(video_files)}")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 User cancelled operation")
            return None

def analyze_video_with_output(video_path=None):
    """
    Analyze video and save results to files
    """
    
    # If no video path provided, let user choose
    if video_path is None:
        video_path = list_available_videos()
        if video_path is None:
            print("❌ No video file selected, program exits")
            return
    
    # Create output directory based on video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create unified main output directory
    main_output_dir = "Output_Files"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create independent subdirectory for each video
    output_dir = os.path.join(main_output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths (based on video name)
    json_output = os.path.join(output_dir, f"emotion_analysis_{video_name}.json")
    csv_output = os.path.join(output_dir, f"emotion_analysis_{video_name}.csv")
    video_output = os.path.join(output_dir, f"analyzed_{video_name}.mp4")
    
    print("=== Starting video emotion analysis and saving results ===")
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Method 1: Frame-by-frame analysis and save to JSON/CSV
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        
        # Get video FPS to calculate accurate timestamps
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        # Calculate analysis interval: every 0.1 seconds
        # For 30fps video: analyze every 3 frames (30fps * 0.1s = 3 frames)
        analysis_interval = max(1, int(fps * 0.1))  # At least every frame
        
        # Calculate max frames for 15 seconds
        max_frames = int(fps * 15)  # 15 seconds of video
        
        print(f"\n📊 Performing emotion analysis every 0.1 seconds (first 15 seconds)...")
        print(f"Video FPS: {fps:.1f}, Analysis interval: every {analysis_interval} frames")
        print(f"Expected analysis points: {max_frames // analysis_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze every 0.1 seconds (every analysis_interval frames)
            if frame_count % analysis_interval == 0:
                try:
                    result = DeepFace.analyze(
                        img_path=frame, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if result:
                        frame_result = {
                            "frame_number": frame_count,
                            "timestamp_seconds": frame_count / fps,  # Use actual FPS
                            "dominant_emotion": result[0]['dominant_emotion'],
                            "emotions": {
                                emotion: float(score) 
                                for emotion, score in result[0]['emotion'].items()
                            }
                        }
                        results.append(frame_result)
                        
                        # Print progress every 30 frames to avoid too much output
                        if len(results) % 10 == 0:
                            print(f"Analysis point {len(results):3d} - Frame {frame_count:4d} ({frame_result['timestamp_seconds']:6.1f}s): {result[0]['dominant_emotion']}")
                        
                except Exception as e:
                    if frame_count % (analysis_interval * 10) == 0:  # Only print errors occasionally
                        print(f"Frame {frame_count} analysis failed: {e}")
            
            frame_count += 1
            
            # Stop after 15 seconds
            if frame_count >= max_frames:
                print(f"✅ Reached 15 seconds limit, processed {frame_count} frames")
                print(f"✅ Total analysis points: {len(results)}")
                break
        
        cap.release()
        
        # Save results to JSON
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                "video_info": {
                    "file_path": video_path,
                    "total_frames_analyzed": len(results),
                    "analysis_date": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # Save results to CSV
        if results:
            df_data = []
            for result in results:
                row = {
                    "frame_number": result["frame_number"],
                    "timestamp_seconds": result["timestamp_seconds"],
                    "dominant_emotion": result["dominant_emotion"]
                }
                # Add all emotion scores
                row.update(result["emotions"])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_output, index=False, encoding='utf-8')
            
            print(f"\n✅ Results saved:")
            print(f"   📄 JSON: {json_output}")
            print(f"   📊 CSV:  {csv_output}")
            print(f"   📈 Analyzed {len(results)} time points")
        
    # Method 2: Use stream function to generate video with analysis results
    print(f"\n🎥 Generating video with analysis results...")
    print(f"Output video: {video_output}")
    
    try:
        # Create database directory
        db_path = "./temp_database"
        os.makedirs(db_path, exist_ok=True)
        
        # Use stream function, specify output path this time
        DeepFace.stream(
            db_path=db_path,
            source=video_path,
            enable_face_analysis=True,
            time_threshold=2,
            frame_threshold=3,
            output_path=video_output  # Specify output video path
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  User stopped video generation")
    except Exception as e:
        print(f"❌ Video generation error: {e}")
    
    print(f"\n🎯 All output files should be in: {os.path.abspath(output_dir)}")

def show_analysis_summary(video_name=None):
    """Display analysis results summary"""
    
    main_output_dir = "Output_Files"
    
    if video_name is None:
        # Search for all video subdirectories in Output_Files directory
        if not os.path.exists(main_output_dir):
            print("❌ Output_Files directory not found")
            return
            
        video_dirs = [d for d in os.listdir(main_output_dir) 
                     if os.path.isdir(os.path.join(main_output_dir, d))]
        
        if not video_dirs:
            print("❌ No analysis results found in Output_Files")
            return
        
        # Find the latest analysis results
        latest_dir = max(video_dirs, 
                        key=lambda x: os.path.getctime(os.path.join(main_output_dir, x)))
        
        json_pattern = os.path.join(main_output_dir, latest_dir, f"emotion_analysis_{latest_dir}.json")
        
        if os.path.exists(json_pattern):
            json_path = json_pattern
        else:
            # If not found by new naming, try to find any json file
            json_files = glob.glob(os.path.join(main_output_dir, latest_dir, "*.json"))
            if not json_files:
                print(f"❌ No analysis results found in {latest_dir}")
                return
            json_path = json_files[0]
    else:
        json_path = os.path.join(main_output_dir, video_name, f"emotion_analysis_{video_name}.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        if results:
            print(f"\n📈 Emotion Analysis Summary - {os.path.basename(json_path)}:")
            print("-" * 60)
            
            # Count main emotions
            emotion_counts = {}
            for result in results:
                emotion = result['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Sort by occurrence count
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            
            total_points = len(results)
            for emotion, count in sorted_emotions:
                percentage = (count / total_points) * 100
                print(f"  {emotion:8s}: {count:3d} times ({percentage:5.1f}%)")
            
            print(f"\nTotal analysis points: {total_points}")
            print(f"Video duration: {results[-1]['timestamp_seconds']:.1f} seconds")
            print(f"📁 Results directory: {os.path.dirname(json_path)}")
    else:
        print(f"❌ Analysis results file does not exist: {json_path}")

def show_all_results_overview():
    """Display overview of all analysis results"""
    
    main_output_dir = "Output_Files"
    
    if not os.path.exists(main_output_dir):
        print("❌ Output_Files directory not found")
        return
    
    video_dirs = [d for d in os.listdir(main_output_dir) 
                 if os.path.isdir(os.path.join(main_output_dir, d))]
    
    if not video_dirs:
        print("❌ No analysis results found in Output_Files")
        return
    
    print(f"\n📁 Output_Files directory overview:")
    print("=" * 70)
    
    for i, video_dir in enumerate(sorted(video_dirs), 1):
        dir_path = os.path.join(main_output_dir, video_dir)
        
        # Count files
        files = os.listdir(dir_path)
        json_files = [f for f in files if f.endswith('.json')]
        csv_files = [f for f in files if f.endswith('.csv')]
        mp4_files = [f for f in files if f.endswith('.mp4')]
        
        print(f"{i}. 📂 {video_dir}/")
        print(f"   📄 JSON: {len(json_files)} files")
        print(f"   📊 CSV:  {len(csv_files)} files") 
        print(f"   🎥 MP4:  {len(mp4_files)} files")
        
        # If there are JSON files, show brief analysis results
        if json_files:
            try:
                json_path = os.path.join(dir_path, json_files[0])
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data['results']
                if results:
                    # Find main emotion
                    emotion_counts = {}
                    for result in results:
                        emotion = result['dominant_emotion']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    top_emotion = max(emotion_counts.items(), key=lambda x: x[1])
                    duration = results[-1]['timestamp_seconds']
                    
                    print(f"   🎯 Main emotion: {top_emotion[0]} ({(top_emotion[1]/len(results)*100):.1f}%)")
                    print(f"   ⏱️  Video duration: {duration:.1f}s")
            except:
                print(f"   ⚠️  Unable to read analysis results")
        
        print()

def generate_emotion_video(input_video_path, output_video_path, emotion_results):
    """
    Generate high-quality playable MP4 video with emotion analysis overlay
    """
    print("🎬 Creating high-quality playable MP4 video with emotion overlay...")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ Cannot open input video")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Input video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Ensure output path has .mp4 extension
    if not output_video_path.lower().endswith('.mp4'):
        base_name = os.path.splitext(output_video_path)[0]
        output_video_path = base_name + '.mp4'
    
    # Try different codec configurations for maximum compatibility
    codec_configs = [
        # Configuration 1: Standard MP4V with specific settings
        {
            'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
            'fps': fps,
            'size': (width, height),
            'description': 'MP4V (most compatible)'
        },
        # Configuration 2: Try with even dimensions (some codecs require this)
        {
            'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
            'fps': fps,
            'size': (width - (width % 2), height - (height % 2)),  # Ensure even dimensions
            'description': 'MP4V with even dimensions'
        },
        # Configuration 3: Lower frame rate for compatibility
        {
            'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
            'fps': 25.0,  # Standard framerate
            'size': (width - (width % 2), height - (height % 2)),
            'description': 'MP4V at 25fps'
        },
        # Configuration 4: XVID as fallback
        {
            'fourcc': cv2.VideoWriter_fourcc(*'XVID'),
            'fps': fps,
            'size': (width - (width % 2), height - (height % 2)),
            'description': 'XVID codec'
        }
    ]
    
    out = None
    used_config = None
    
    for i, config in enumerate(codec_configs):
        try:
            print(f"🔧 Trying configuration {i+1}: {config['description']}")
            
            out = cv2.VideoWriter(
                output_video_path, 
                config['fourcc'], 
                config['fps'], 
                config['size']
            )
            
            if out.isOpened():
                # Test write a frame to make sure it really works
                test_frame = cv2.imread(input_video_path) if os.path.exists(input_video_path + '.jpg') else None
                if test_frame is None:
                    # Create a test frame
                    test_frame = cv2.resize(cv2.imread('temp_frame.jpg', cv2.IMREAD_COLOR) if os.path.exists('temp_frame.jpg') else 
                                          (cv2.cvtColor(cv2.imread(input_video_path) if os.path.exists(input_video_path + '.frame') else 
                                                       255 * cv2.ones((config['size'][1], config['size'][0], 3), dtype=cv2.uint8), cv2.COLOR_BGR2RGB)), config['size'])
                
                if test_frame is None:
                    # Create a simple test frame
                    test_frame = 255 * cv2.ones((config['size'][1], config['size'][0], 3), dtype=cv2.uint8)
                else:
                    test_frame = cv2.resize(test_frame, config['size'])
                
                try:
                    out.write(test_frame)
                    used_config = config
                    print(f"✅ Successfully initialized with: {config['description']}")
                    break
                except:
                    out.release()
                    out = None
                    print(f"❌ Test write failed for: {config['description']}")
            else:
                if out:
                    out.release()
                    out = None
                print(f"❌ Failed to open with: {config['description']}")
                
        except Exception as e:
            if out:
                out.release()
                out = None
            print(f"❌ Error with {config['description']}: {e}")
    
    if out is None or used_config is None:
        print("❌ Failed to initialize video writer with any configuration")
        cap.release()
        return None
    
    # Now process the actual video
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Create emotion results lookup for faster access
    emotion_lookup = {}
    for result in emotion_results:
        emotion_lookup[result['frame_number']] = result
    
    # Calculate how many frames to process (15 seconds)
    max_frames_to_process = int(used_config['fps'] * 15)
    frames_to_process = min(total_frames, max_frames_to_process)
    
    print(f"Processing {frames_to_process} frames at {used_config['fps']:.1f}fps...")
    
    frames_written = 0
    
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to match output configuration
        if frame.shape[:2] != (used_config['size'][1], used_config['size'][0]):
            frame = cv2.resize(frame, used_config['size'])
        
        # Add emotion overlay if we have analysis for this frame
        if frame_count in emotion_lookup:
            result = emotion_lookup[frame_count]
            emotion = result['dominant_emotion']
            timestamp = result['timestamp_seconds']
            
            # Create overlay with better contrast and readability
            overlay_frame = frame.copy()
            
            # Main emotion text
            main_text = f"Emotion: {emotion.upper()}"
            time_text = f"Time: {timestamp:.1f}s"
            
            # Font settings for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(1.0, used_config['size'][0] / 800)  # Scale based on video width
            thickness = max(2, int(font_scale * 2))
            
            # Calculate text sizes
            main_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
            time_size = cv2.getTextSize(time_text, font, font_scale * 0.8, thickness)[0]
            
            # Background dimensions
            padding = 15
            bg_width = max(main_size[0], time_size[0]) + 2 * padding
            bg_height = main_size[1] + time_size[1] + 4 * padding
            
            # Draw semi-transparent background
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, overlay_frame, 0.2, 0, overlay_frame)
            
            # Add white border for better visibility
            cv2.rectangle(overlay_frame, (8, 8), (12 + bg_width, 12 + bg_height), (255, 255, 255), 2)
            
            # Add main emotion text (bright green)
            cv2.putText(overlay_frame, main_text, (15, 35), 
                       font, font_scale, (0, 255, 0), thickness)
            
            # Add timestamp (white)
            cv2.putText(overlay_frame, time_text, (15, 35 + main_size[1] + 15), 
                       font, font_scale * 0.8, (255, 255, 255), thickness)
            
            frame = overlay_frame
        
        # Write frame to output video
        try:
            out.write(frame)
            frames_written += 1
        except Exception as e:
            print(f"❌ Error writing frame {frame_count}: {e}")
            break
        
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / frames_to_process) * 100
            print(f"Video progress: {progress:.1f}% ({frame_count}/{frames_to_process}) - {frames_written} frames written")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"✅ Video processing complete: {frames_written} frames written")
    
    # Verify output file
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"📦 Output file size: {file_size:.1f}MB")
        
        # Verify the video can be read
        test_cap = cv2.VideoCapture(output_video_path)
        if test_cap.isOpened():
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_fps = test_cap.get(cv2.CAP_PROP_FPS)
            test_duration = test_frames / test_fps if test_fps > 0 else 0
            test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Video verification successful:")
            print(f"   📊 Resolution: {test_width}x{test_height}")
            print(f"   🎬 Frames: {test_frames}")
            print(f"   ⏱️  FPS: {test_fps:.1f}")
            print(f"   ⏲️  Duration: {test_duration:.1f}s")
            
            # Test reading first frame
            ret, test_frame = test_cap.read()
            if ret:
                print(f"   ✅ First frame readable: {test_frame.shape}")
            else:
                print(f"   ⚠️  Cannot read first frame")
            
            test_cap.release()
            
            return output_video_path
        else:
            print("❌ Output video exists but cannot be opened")
            return None
    else:
        print("❌ Failed to create output video file")
        return None

if __name__ == "__main__":
    print("🎬 DeepFace Video Emotion Analysis Tool")
    print("=" * 50)
    
    # Show existing results overview
    show_all_results_overview()
    
    # Analyze video (will automatically prompt for selection)
    analyze_video_with_output()
    
    # Show analysis summary
    show_analysis_summary()
    
    # Show updated overview again
    print("\n" + "="*50)
    print("🔄 Updated results overview:")
    show_all_results_overview()#display all results overview after analysis

