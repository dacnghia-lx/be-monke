import cv2
import mediapipe as mp
import math


# --- 1. Khởi tạo MediaPipe và các đối tượng cần thiết ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- 2. Tải và chuẩn bị tất cả các ảnh ---
image_mouth_gesture = cv2.imread('khi_mieng.png')
image_point_gesture = cv2.imread('khi_chidu.png')
# Tải ảnh mặc định bạn vừa gửi
image_default = cv2.imread('khi_ngu.png') 

# Kiểm tra xem ảnh đã được tải thành công chưa
if image_mouth_gesture is None or image_point_gesture is None or image_default is None:
    print("Lỗi: Không thể tải một hoặc nhiều ảnh. Hãy chắc chắn rằng tên file và đường dẫn là chính xác.")

# Thay đổi kích thước ảnh để hiển thị vừa vặn
image_mouth_gesture = cv2.resize(image_mouth_gesture, (500, 500))
image_point_gesture = cv2.resize(image_point_gesture, (500, 500))
image_default = cv2.resize(image_default, (500, 500))

cap = cv2.VideoCapture(0)
cap.set(3, 500)  # Chiều rộng
cap.set(4, 500)  # Chiều cao

# --- 3. Vòng lặp xử lý chính ---
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands, \
    mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("wtf.")
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        hand_results = hands.process(image_rgb)
        face_results = face_mesh.process(image_rgb)
        
        detected_gesture = None 

        # Gesture recognition
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # A. Gesture mouth
                if face_results.multi_face_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_finger_tip.x * image_width), int(index_finger_tip.y * image_height)
                    for face_landmarks in face_results.multi_face_landmarks:
                        mouth_top = face_landmarks.landmark[13]
                        mx_top, my_top = int(mouth_top.x * image_width), int(mouth_top.y * image_height)
                        if math.sqrt((ix - mx_top)**2 + (iy - my_top)**2) < 50:
                            detected_gesture = "mouth"
                            break
                    if detected_gesture: break

                # B. Nhận diện cử chỉ giơ ngón trỏ
                if not detected_gesture:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                    if index_tip.y < index_pip.y and middle_tip.y > middle_pip.y:
                        detected_gesture = "point"

        # --- 4. Hiển thị ảnh con khỉ tương ứng ---
        if detected_gesture == "mouth":
            cv2.imshow('Monkey', image_mouth_gesture)
        elif detected_gesture == "point":
            cv2.imshow('Monkey', image_point_gesture)
        else:
            # Nếu không phát hiện cử chỉ nào, hiển thị ảnh mặc định
            cv2.imshow('Monkey', image_default)

        # Hiển thị khung hình từ camera
        cv2.imshow('Camera', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
