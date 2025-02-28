# Trong đoạn code sau, mô hình sử dụng là mô hình học sâu QuickDraw. Mô hình này được huấn luyện để nhận diện các hình vẽ theo cách mà người dùng vẽ bằng tay.
# Mô hình này là một mạng nơ-ron tích chập (CNN) được huấn luyện trên tập dữ liệu QuickDraw, một bộ dữ liệu lớn chứa các hình vẽ được vẽ bởi người dùng trên internet.

# Cấu trúc mô hình
# Mô hình này là một mạng CNN gồm các thành phần chính sau:

# Lớp Convolution (Conv1 và Conv2):
# Lớp tích chập đầu tiên (conv1) có 32 filters, mỗi filter có kích thước 5x5. Sau đó, sử dụng hàm kích hoạt ReLU và MaxPooling.
# Lớp tích chập thứ hai (conv2) có 64 filters, cũng sử dụng ReLU và MaxPooling.

# Lớp Fully Connected (FC):
# Lớp FC đầu tiên có 512 đơn vị với Dropout 50%.
# Lớp FC thứ hai có 128 đơn vị với Dropout 50%.
# Lớp cuối cùng dự đoán một trong các lớp đầu ra, với số lượng lớp bằng số đối tượng trong tập dữ liệu (15 lớp trong ví dụ của bạn).

# Tập Dữ Liệu Huấn Luyện
# Mô hình được huấn luyện trên tập dữ liệu QuickDraw, tập dữ liệu này chứa các hình vẽ của nhiều đối tượng khác nhau do người dùng vẽ. Các đối tượng trong CLASSES là những đối tượng mà
# mô hình có thể nhận diện như apple, book, tree, v.v.
# Tập dữ liệu này có các hình vẽ được vẽ bằng tay, do đó mô hình học cách phân loại những hình vẽ đó thành các lớp tương ứng.


import cv2
from collections import deque
import mediapipe as mp
import numpy as np
from src.utils import get_images, get_overlay
from src.config import *
import torch


# mp_drawing: Đối tượng tiện ích để vẽ các điểm và đường nối trên ảnh.
# mp_drawing_styles: Chứa các kiểu dáng cho các điểm và đường nối.
# mp_hands: Đối tượng nhận diện bàn tay từ MediaPipe.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# if torch.cuda.is_available(): Kiểm tra xem có GPU CUDA sẵn có không.
# model = torch.load(...): Tải mô hình đã được huấn luyện từ file.
# model.eval(): Đặt mô hình vào chế độ đánh giá (không cập nhật trọng số).
# predicted_class = None: Khởi tạo biến để lưu lớp dự đoán.
if torch.cuda.is_available():
    model = torch.load("trained_models/whole_model_quickdraw")
else:
    model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
model.eval()
predicted_class = None


# cap = cv2.VideoCapture(0): Mở camera để đọc video (0 là chỉ số camera mặc định).
# points = deque(maxlen=512): Tạo một deque để lưu trữ các điểm vẽ, với độ dài tối đa là 512.
# canvas = np.zeros((480, 640, 3), dtype=np.uint8): Tạo một canvas trống để vẽ, có kích thước 480x640 và 3 kênh màu.
# is_drawing = False: Biến để theo dõi trạng thái vẽ.
# is_shown = False: Biến để theo dõi trạng thái hiển thị.
# class_images = get_images("images", CLASSES): Lấy hình ảnh liên quan đến các lớp từ thư mục "images".
cap = cv2.VideoCapture(0)
points = deque(maxlen=512)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
is_drawing = False
is_shown = False
class_images = get_images("images", CLASSES)


# Khởi tạo đối tượng Hands từ MediaPipe với các tham số:
# max_num_hands=1: Tối đa chỉ nhận diện một bàn tay.
# model_complexity=0: Độ phức tạp của mô hình, 0 là sử dụng mô hình nhanh.
# min_detection_confidence=0.5: Ngưỡng độ tin cậy tối thiểu để phát hiện bàn tay.
# min_tracking_confidence=0.5: Ngưỡng độ tin cậy tối thiểu để theo dõi bàn tay.
with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    
    # while cap.isOpened(): Vòng lặp chạy cho đến khi camera đóng.
    # success, image = cap.read(): Đọc khung hình từ camera.
    # image = cv2.flip(image, 1): Lật hình ảnh theo chiều ngang (để giống như gương).
    # if not success: continue: Nếu không đọc được khung hình, tiếp tục vòng lặp.
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            continue


        # image.flags.writeable = False: Đánh dấu hình ảnh là không thể ghi (để cải thiện hiệu suất).
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB): Chuyển đổi màu sắc từ BGR sang RGB.
        # results = hands.process(image): Xử lý hình ảnh để phát hiện các điểm trên bàn tay.
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)


        # image.flags.writeable = True: Đánh dấu hình ảnh lại là có thể ghi.
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR): Chuyển đổi lại màu sắc từ RGB về BGR để hiển thị.
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        # if results.multi_hand_landmarks: Kiểm tra xem có điểm nào của bàn tay được phát hiện không.
        # for hand_landmarks in results.multi_hand_landmarks: Duyệt qua từng bàn tay được phát hiện.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                
                # Kiểm tra xem ngón tay cái, ngón trỏ và ngón giữa có đang ở vị trí thấp hơn các ngón khác không. Điều này giúp xác định xem người dùng có đang vẽ hay không. 
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                        hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                    
                    
                    # Nếu có điểm vẽ (len(points)), thay đổi trạng thái is_drawing và is_shown.
                    # canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY): Chuyển canvas sang màu xám.
                    # canvas_gs = cv2.medianBlur(canvas_gs, 9) và canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0): Áp dụng làm mờ cho canvas.
                    # ys, xs = np.nonzero(canvas_gs): Lấy tọa độ của các điểm không phải là 0 trên canvas.
                    # Nếu có tọa độ, cắt và thay đổi kích thước hình ảnh được vẽ thành 28x28 pixel.
                    # cropped_image = torch.from_numpy(cropped_image): Chuyển đổi hình ảnh cắt thành tensor PyTorch.
                    # logits = model(cropped_image): Dự đoán lớp dựa trên hình ảnh.
                    # predicted_class = torch.argmax(logits[0]): Lấy lớp có xác suất cao nhất.
                    # Đặt lại points và canvas về trạng thái mặc định.
                    if len(points):
                        is_drawing = False
                        is_shown = True
                        canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        canvas_gs = cv2.medianBlur(canvas_gs, 9)
                        canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)
                        ys, xs = np.nonzero(canvas_gs)
                        if len(ys) and len(xs):
                            min_y = np.min(ys)
                            max_y = np.max(ys)
                            min_x = np.min(xs)
                            max_x = np.max(xs)
                            cropped_image = canvas_gs[min_y:max_y, min_x: max_x]
                            cropped_image = cv2.resize(cropped_image, (28, 28))
                            cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :]
                            cropped_image = torch.from_numpy(cropped_image)
                            logits = model(cropped_image)
                            predicted_class = torch.argmax(logits[0])
                            points = deque(maxlen=512)
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                            
                            
                # Nếu không vẽ, thay đổi trạng thái để cho biết người dùng đang vẽ.
                # Thêm tọa độ của ngón tay vào points.
                # Vẽ đường nối giữa các điểm trên cả image và canvas.
                else:
                    is_drawing = True
                    is_shown = False
                    points.append((int(hand_landmarks.landmark[8].x*640), int(hand_landmarks.landmark[8].y*480)))
                    for i in range(1, len(points)):
                        cv2.line(image, points[i - 1], points[i], (0, 255, 0), 2)
                        cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 5)
                        
                        
                # Vẽ các điểm trên bàn tay lên hình ảnh.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                
                # Nếu không vẽ và đã hiển thị hình ảnh, hiển thị văn bản "You are drawing" trên hình ảnh.
                # Chồng hình ảnh dự đoán lên hình ảnh chính.
                if not is_drawing and is_shown:
                    cv2.putText(image, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 5,
                                cv2.LINE_AA)
                    image[5:65, 490:550] = get_overlay(image[5:65, 490:550], class_images[predicted_class], (60, 60))


        # Hiển thị hình ảnh đã xử lý.
        # Dừng vòng lặp nếu phím Esc (27) được nhấn.
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        

# Giải phóng camera khi kết thúc chương trình.
cap.release()
