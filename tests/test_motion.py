import sys; sys.path.insert(0, 'src')
from camera import Camera
from preprocess import FrameAnalyzer
a = FrameAnalyzer()
print("请挥挥手或摇头...")
with Camera() as cam:
    for _ in range(5): cam.capture()
    for i in range(60):
        r = a.analyze_frame(cam.capture())
        m = r['motion']
        if m['motion_ratio'] > 0:
            print(f'帧{i+1}: ratio={m["motion_ratio"]:.6f} level={m["level"]} regions={m["num_regions"]}')
            break
    else:
        print('60帧全部归零——MOG2未检测到运动')
