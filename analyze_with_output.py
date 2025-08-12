from deepface import DeepFace
import cv2
import os
import json
import pandas as pd
from datetime import datetime
import glob

def list_available_videos():
    """列出所有可用的视频文件并让用户选择"""
    
    video_dir = r"C:\MyDeepFace\InputVideos"
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    # 找到所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        print("❌ 在 InputVideos 目录中没有找到视频文件")
        return None
    
    print("\n📹 发现以下视频文件:")
    print("=" * 60)
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"{i}. {filename}")
            print(f"   📐 分辨率: {width}x{height}")
            print(f"   ⏱️  时长: {duration:.1f}秒")
            print(f"   📦 大小: {size_mb:.1f}MB")
            print()
        else:
            print(f"{i}. {filename} (无法读取视频信息)")
            print()
    
    # 用户选择
    while True:
        try:
            choice = input(f"请选择要分析的视频 (1-{len(video_files)}) 或输入 'q' 退出: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(video_files):
                selected_video = video_files[choice_num - 1]
                print(f"\n✅ 您选择了: {os.path.basename(selected_video)}")
                return selected_video
            else:
                print(f"❌ 请输入 1-{len(video_files)} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n👋 用户取消操作")
            return None

def analyze_video_with_output(video_path=None):
    """
    分析视频并保存结果到文件
    """
    
    # 如果没有提供视频路径，让用户选择
    if video_path is None:
        video_path = list_available_videos()
        if video_path is None:
            print("❌ 没有选择视频文件，程序退出")
            return
    
    # 基于视频文件名创建输出目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建统一的主输出目录
    main_output_dir = "Output_Files"
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 为每个视频创建独立的子目录
    output_dir = os.path.join(main_output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径（基于视频名称）
    json_output = os.path.join(output_dir, f"emotion_analysis_{video_name}.json")
    csv_output = os.path.join(output_dir, f"emotion_analysis_{video_name}.csv")
    video_output = os.path.join(output_dir, f"analyzed_{video_name}.mp4")
    
    print("=== 开始视频情感分析并保存结果 ===")
    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")
    
    # 方法1: 逐帧分析并保存到 JSON/CSV
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        
        print("\n📊 正在进行逐帧分析...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔30帧分析一次（约1秒一次，30fps视频）
            if frame_count % 30 == 0:
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
                            "timestamp_seconds": frame_count / 30.0,  # 假设30fps
                            "dominant_emotion": result[0]['dominant_emotion'],
                            "emotions": {
                                emotion: float(score) 
                                for emotion, score in result[0]['emotion'].items()
                            }
                        }
                        results.append(frame_result)
                        
                        print(f"帧 {frame_count:4d} ({frame_result['timestamp_seconds']:6.1f}s): {result[0]['dominant_emotion']}")
                        
                except Exception as e:
                    print(f"帧 {frame_count} 分析失败: {e}")
            
            frame_count += 1
            
            # 限制处理帧数（可选）
            if frame_count > 3000:  # 处理前100秒
                break
        
        cap.release()
        
        # 保存结果到 JSON
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                "video_info": {
                    "file_path": video_path,
                    "total_frames_analyzed": len(results),
                    "analysis_date": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # 保存结果到 CSV
        if results:
            df_data = []
            for result in results:
                row = {
                    "frame_number": result["frame_number"],
                    "timestamp_seconds": result["timestamp_seconds"],
                    "dominant_emotion": result["dominant_emotion"]
                }
                # 添加所有情感分数
                row.update(result["emotions"])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_output, index=False, encoding='utf-8')
            
            print(f"\n✅ 结果已保存:")
            print(f"   📄 JSON: {json_output}")
            print(f"   📊 CSV:  {csv_output}")
            print(f"   📈 分析了 {len(results)} 个时间点")
        
    # 方法2: 使用 stream 函数生成带分析结果的视频
    print(f"\n🎥 正在生成带分析结果的视频...")
    print(f"输出视频: {video_output}")
    
    try:
        # 创建数据库目录
        db_path = "./temp_database"
        os.makedirs(db_path, exist_ok=True)
        
        # 使用 stream 函数，这次指定输出路径
        DeepFace.stream(
            db_path=db_path,
            source=video_path,
            enable_face_analysis=True,
            time_threshold=2,
            frame_threshold=3,
            output_path=video_output  # 指定输出视频路径
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  用户停止了视频生成")
    except Exception as e:
        print(f"❌ 视频生成出错: {e}")
    
    print(f"\n🎯 所有输出文件应该在: {os.path.abspath(output_dir)}")

def show_analysis_summary(video_name=None):
    """显示分析结果摘要"""
    
    main_output_dir = "Output_Files"
    
    if video_name is None:
        # 查找 Output_Files 目录中的所有视频子目录
        if not os.path.exists(main_output_dir):
            print("❌ 没有找到 Output_Files 目录")
            return
            
        video_dirs = [d for d in os.listdir(main_output_dir) 
                     if os.path.isdir(os.path.join(main_output_dir, d))]
        
        if not video_dirs:
            print("❌ 在 Output_Files 中没有找到分析结果")
            return
        
        # 找到最新的分析结果
        latest_dir = max(video_dirs, 
                        key=lambda x: os.path.getctime(os.path.join(main_output_dir, x)))
        
        json_pattern = os.path.join(main_output_dir, latest_dir, f"emotion_analysis_{latest_dir}.json")
        
        if os.path.exists(json_pattern):
            json_path = json_pattern
        else:
            # 如果按新命名找不到，尝试找任何json文件
            json_files = glob.glob(os.path.join(main_output_dir, latest_dir, "*.json"))
            if not json_files:
                print(f"❌ 在 {latest_dir} 中没有找到分析结果")
                return
            json_path = json_files[0]
    else:
        json_path = os.path.join(main_output_dir, video_name, f"emotion_analysis_{video_name}.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        if results:
            print(f"\n📈 情感分析摘要 - {os.path.basename(json_path)}:")
            print("-" * 60)
            
            # 统计主要情感
            emotion_counts = {}
            for result in results:
                emotion = result['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 按出现次数排序
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            
            total_points = len(results)
            for emotion, count in sorted_emotions:
                percentage = (count / total_points) * 100
                print(f"  {emotion:8s}: {count:3d} 次 ({percentage:5.1f}%)")
            
            print(f"\n总分析点数: {total_points}")
            print(f"视频时长: {results[-1]['timestamp_seconds']:.1f} 秒")
            print(f"📁 结果目录: {os.path.dirname(json_path)}")
    else:
        print(f"❌ 分析结果文件不存在: {json_path}")

def show_all_results_overview():
    """显示所有分析结果的概览"""
    
    main_output_dir = "Output_Files"
    
    if not os.path.exists(main_output_dir):
        print("❌ 没有找到 Output_Files 目录")
        return
    
    video_dirs = [d for d in os.listdir(main_output_dir) 
                 if os.path.isdir(os.path.join(main_output_dir, d))]
    
    if not video_dirs:
        print("❌ 在 Output_Files 中没有找到分析结果")
        return
    
    print(f"\n📁 Output_Files 目录概览:")
    print("=" * 70)
    
    for i, video_dir in enumerate(sorted(video_dirs), 1):
        dir_path = os.path.join(main_output_dir, video_dir)
        
        # 统计文件
        files = os.listdir(dir_path)
        json_files = [f for f in files if f.endswith('.json')]
        csv_files = [f for f in files if f.endswith('.csv')]
        mp4_files = [f for f in files if f.endswith('.mp4')]
        
        print(f"{i}. 📂 {video_dir}/")
        print(f"   📄 JSON: {len(json_files)} 个")
        print(f"   📊 CSV:  {len(csv_files)} 个") 
        print(f"   🎥 MP4:  {len(mp4_files)} 个")
        
        # 如果有JSON文件，显示简要分析结果
        if json_files:
            try:
                json_path = os.path.join(dir_path, json_files[0])
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data['results']
                if results:
                    # 找出主要情感
                    emotion_counts = {}
                    for result in results:
                        emotion = result['dominant_emotion']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    top_emotion = max(emotion_counts.items(), key=lambda x: x[1])
                    duration = results[-1]['timestamp_seconds']
                    
                    print(f"   🎯 主要情感: {top_emotion[0]} ({(top_emotion[1]/len(results)*100):.1f}%)")
                    print(f"   ⏱️  视频时长: {duration:.1f}秒")
            except:
                print(f"   ⚠️  无法读取分析结果")
        
        print()

if __name__ == "__main__":
    print("🎬 DeepFace 视频情感分析工具")
    print("=" * 50)
    
    # 显示现有结果概览
    show_all_results_overview()
    
    # 分析视频（会自动提示选择）
    analyze_video_with_output()
    
    # 显示分析摘要
    show_analysis_summary()
    
    # 再次显示更新后的概览
    print("\n" + "="*50)
    print("🔄 更新后的结果概览:")
    show_all_results_overview()#display all results overview after analysis

