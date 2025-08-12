"""
如果您想启用人脸识别功能，可以按以下步骤操作：

1. 创建人脸数据库目录结构：
   database/
   ├── person1/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── person2/
       ├── image1.jpg
       └── image2.jpg

2. 然后修改代码：
"""

import os
from deepface import DeepFace

def setup_face_database():
    """设置人脸数据库示例"""
    
    # 创建数据库目录结构
    db_path = "face_database"
    
    # 创建示例目录
    people = ["person1", "person2", "unknown"]
    
    for person in people:
        person_dir = os.path.join(db_path, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # 创建说明文件
        readme_path = os.path.join(person_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"请将 {person} 的照片放在这个目录中\n")
            f.write("支持的格式: .jpg, .jpeg, .png\n")
            f.write("建议: 每人2-5张不同角度的清晰照片\n")
    
    print(f"✅ 人脸数据库目录已创建: {os.path.abspath(db_path)}")
    print("📝 请在各个子目录中放入对应人员的照片")
    
    return db_path

def analyze_with_face_recognition():
    """带人脸识别的视频分析"""
    
    # 设置数据库
    db_path = setup_face_database()
    
    video_path = r"C:\MyDeepFace\InputVideos\102-30-640x360.mp4"
    output_video = "output_results/video_with_face_recognition.mp4"
    
    print("如果您的数据库中有照片，运行以下代码:")
    print(f"""
    DeepFace.stream(
        db_path="{db_path}",
        source="{video_path}",
        enable_face_analysis=True,
        output_path="{output_video}"
    )
    """)

if __name__ == "__main__":
    setup_face_database()
