from deepface import DeepFace
import cv2
import os

def test_video_emotion_analysis():
    """
    测试 DeepFace 对本地视频文件的情感识别功能
    """
    
    # 示例1: 使用 analyze 函数分析视频帧
    print("=== 方法1: 手动读取视频帧进行分析 ===")
    
    # 使用您提供的实际视频文件
    video_path = r"C:\MyDeepFace\InputVideos\102-30-640x360.mp4"
    
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔30帧分析一次（减少计算量）
            if frame_count % 30 == 0:
                try:
                    # 使用 DeepFace 分析当前帧
                    result = DeepFace.analyze(
                        img_path=frame, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if result:
                        emotion_info = result[0]['emotion']
                        dominant_emotion = result[0]['dominant_emotion']
                        
                        print(f"帧 {frame_count}: 主要情感 = {dominant_emotion}")
                        print(f"情感分析详情: {emotion_info}")
                        
                except Exception as e:
                    print(f"帧 {frame_count} 分析失败: {e}")
            
            frame_count += 1
            
            # 只处理前几帧作为示例
            if frame_count > 150:
                break
        
        cap.release()
    else:
        print(f"视频文件 {video_path} 不存在")
    
    print("\n=== 方法2: 使用 stream 函数分析视频文件 ===")
    
    # 示例2: 使用 stream 函数（适用于实时分析）
    # 注意：stream 函数会打开一个窗口显示视频分析结果
    if os.path.exists(video_path):
        print("开始实时视频分析...")
        print("按 'q' 键退出窗口")
        
        try:
            # 创建数据库目录（可以是空的）
            db_path = "./temp_database"
            os.makedirs(db_path, exist_ok=True)
            
            # 使用 stream 函数分析视频文件
            DeepFace.stream(
                db_path=db_path,       # 数据库路径（可以是空目录）
                source=video_path,     # 您的视频文件路径
                enable_face_analysis=True,
                time_threshold=2,      # 显示结果的时间（秒）
                frame_threshold=3      # 检测帧数阈值
            )
        except Exception as e:
            print(f"Stream 分析出错: {e}")
    else:
        print(f"视频文件不存在: {video_path}")

def test_emotion_with_sample_image():
    """
    使用示例图片测试情感识别功能
    """
    print("\n=== 测试图片情感识别 ===")
    
    # 创建一个简单的测试
    import numpy as np
    
    # 创建一个示例图像（实际使用时应该是真实图片）
    # 这里只是演示，实际需要真实的人脸图片
    try:
        # 如果您有图片文件，可以这样使用：
        # result = DeepFace.analyze(img_path="path/to/your/image.jpg", actions=['emotion'])
        
        print("DeepFace 支持以下情感识别：")
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for emotion in emotions:
            print(f"- {emotion}")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    print("DeepFace 视频情感识别测试")
    print("=" * 50)
    
    test_emotion_with_sample_image()
    test_video_emotion_analysis()
