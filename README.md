# TurtleBot 4 — Real-time Facial Emotion Recognition & Kinematic Control
## Nhận Diện Biểu Cảm Khuôn Mặt và Điều Khiển Robot TurtleBot 4

Package ROS 2 Python điều khiển robot **TurtleBot 4** (thực tế hoặc mô phỏng trên **Gazebo Fortress**) theo các quỹ đạo động học tương ứng với biểu cảm khuôn mặt qua webcam bằng kỹ thuật thị giác máy tính (**OpenCV** & **DeepFace**).

---

## 1. Bảng Ánh Xạ Biểu Cảm & Quỹ Đạo Robot

Hệ thống nhận diện **6 trạng thái biểu cảm** và điều khiển đế vi sai của TurtleBot 4 qua topic `/cmd_vel`:

| STT | Biểu cảm khuôn mặt | Nhãn topic `/face_emotion` | Vận tốc $v_x$ (m/s) | Vận tốc $\omega_z$ (rad/s) | Quỹ đạo chuyển động của TurtleBot 4 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Happy (Vui vẻ)** | `happy` | $+0.15$ | $+0.40$ | **Di chuyển theo quỹ đạo cong sang trái** (tiến tới đồng thời rẽ trái) |
| **2** | **Sad (Buồn)** | `sad` | $+0.15$ | $-0.40$ | **Di chuyển theo quỹ đạo cong sang phải** (tiến tới đồng thời rẽ phải) |
| **3** | **Neutral (Bình thường)** | `neutral` | $0.00$ | $0.00$ | **Dừng tại chỗ** (robot đứng yên an toàn) |
| **4** | **Angry (Tức giận)** | `angry` | $-0.10$ | $0.00$ | **Lùi nhẹ về phía sau** (chuyển động lùi thẳng êm ái) |
| **5** | **Surprise (Ngạc nhiên)** | `surprise` | $0.00$ | $+0.60$ | **Xoay một vòng tại chỗ ($360^\circ$)** (quay $2\pi$ rad rồi tự động hãm dừng) |
| **6** | **Fear (Sợ hãi)** | `fear` | $+0.10$ | $0.00$ | **Tiến nhẹ về phía trước** (chuyển động tiến thẳng êm ái) |

---

## 2. Cấu Trúc Thư Mục Package

```text
robot_face_emotion_control/
├── CMakeLists.txt / package.xml        # ROS 2 Package metadata & dependencies
├── setup.py / setup.cfg                # Cấu hình cài đặt thực thi Python
├── requirements.txt                    # Thư viện Python (DeepFace, OpenCV, TensorFlow)
├── .gitignore                          # Cấu hình bỏ qua các file build/cache
├── README.md                           # Hướng dẫn chi tiết cài đặt và vận hành
├── test_webcam_emotion.py              # Script test nhanh webcam không cần Gazebo
├── config/
│   └── emotion_params.yaml             # Cấu hình tham số camera, ngưỡng tin cậy & động học
├── launch/
│   └── face_emotion_control.launch.py  # Khởi chạy đồng thời Node thị giác và Node điều khiển
├── resource/
│   └── robot_face_emotion_control      # Resource marker
├── robot_face_emotion_control/
│   ├── __init__.py
│   ├── face_emotion_detector.py        # DeepFace adapter & tự nhận diện môi trường ảo
│   ├── emotion_logic.py                # Bộ lọc debouncer StableEmotion & từ điển song ngữ
│   ├── face_emotion_node.py            # Node camera 30 FPS, đa luồng & giao diện HUD
│   └── emotion_behavior_node.py        # Node FSM, bộ lọc gia tốc, Watchdog & /cmd_vel
└── test/
    └── test_face_emotion.py            # Bộ unit test tự động kiểm thử logic và động học
```

---

## 3. Cài Đặt Môi Trường

### Yêu cầu hệ thống:
- Hệ điều hành: Ubuntu 22.04 LTS
- ROS 2: Humble Hawksbill
- Python: 3.10

### Thiết lập môi trường ảo Python:
```bash
# Tạo môi trường ảo hỗ trợ ROS site-packages
python3 -m venv --system-site-packages .venv-face
source .venv-face/bin/activate

# Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Hướng Dẫn Biên Dịch & Chạy

### Bước 1: Build package trong ROS 2 Workspace
```bash
cd ~/workspaces/turtlebot4_ws  # hoặc đường dẫn workspace của bạn
source /opt/ros/humble/setup.bash
source .venv-face/bin/activate

colcon build --symlink-install --packages-select robot_face_emotion_control
source install/setup.bash
```

---

### Bước 2: Kiểm tra nhanh Webcam & DeepFace (Không cần mở Gazebo)
Chạy script kiểm tra độc lập để xác nhận camera và DeepFace nhận diện tốt:
```bash
python3 test_webcam_emotion.py
```
*(Nhấn `q` trên cửa sổ camera để thoát).*

---

### Bước 3: Khởi chạy Toàn bộ Hệ thống Điều khiển TurtleBot 4

Mở 3 Terminal riêng biệt:

#### Terminal 1: Mở mô phỏng Gazebo (nếu dùng robot ảo)
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=69
ros2 launch robot_sim sim.launch.py enable_camera:=true
```

#### Terminal 2: Khởi chạy Node Nhận diện & Node Điều khiển
```bash
source /opt/ros/humble/setup.bash
source .venv-face/bin/activate
source install/setup.bash
export ROS_DOMAIN_ID=69

ros2 launch robot_face_emotion_control face_emotion_control.launch.py
```
> *Tùy chọn*: Thêm `camera_id:=1` nếu dùng webcam rời ngoài, hoặc `show_window:=false` nếu chạy headless.

#### Terminal 3: Giám sát Topic thời gian thực
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=69

# Xem vận tốc gửi vào robot:
ros2 topic echo /cmd_vel

# Xem biểu cảm nhận diện được:
ros2 topic echo /face_emotion

# Xem trạng thái điều khiển FSM:
ros2 topic echo /emotion_control_state
```

---

## 5. Phím Tắt Điều Khiển trên Cửa Sổ Webcam

- **`e`**: Kích hoạt **Emergency Stop (Phanh khẩn cấp)** ngay lập tức. Robot dừng đứng yên và khóa chuyển động.
- **`r`**: **Release (Nhả phanh)** để tiếp tục nhận diện và điều khiển bình thường.
- **`q`**: **Thoát chương trình** an toàn (tự động gửi gói lệnh vận tốc $\vec{v} = 0, \vec{\omega} = 0$ trước khi đóng).

---
