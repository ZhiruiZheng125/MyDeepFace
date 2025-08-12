import cv2
import os

def check_video_file():
    """检查视频文件是否可以正常读取"""
    video_path = r"C:\MyDeepFace\InputVideos\102-30-640x360.mp4"
    
    print(f"检查视频文件: {video_path}")
    print("-" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print("❌ 文件不存在！")
        return False
    
    print("✅ 文件存在")
    print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # 尝试打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ 无法打开视频文件！")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print("✅ 视频文件信息:")
    print(f"   - 分辨率: {width}x{height}")
    print(f"   - 帧率: {fps:.2f} FPS")
    print(f"   - 总帧数: {frame_count}")
    print(f"   - 时长: {duration:.2f} 秒")
    
    # 读取第一帧测试
    ret, frame = cap.read()
    if ret:
        print("✅ 成功读取第一帧")
    else:
        print("❌ 无法读取视频帧")
        cap.release()
        return False
    
    cap.release()
    return True

if __name__ == "__main__":
    check_video_file()
