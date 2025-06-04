"""
Chương trình nhận diện trái cây sử dụng YOLOv11 và webcam
Sử dụng thư viện ultralytics phiên bản mới nhất
"""
import cv2
import time
import argparse
from ultralytics import YOLO
import numpy as np

# Từ điển ánh xạ tên trái cây từ tiếng Anh sang tiếng Việt
FRUIT_NAMES_VI = {
	'Apple': 'Táo',
	'Banana': 'Chuối',
	'Grapes': 'Nho',
	'Kiwi': 'Kiwi',
	'Mango': 'Xoài',
	'Orange': 'Cam',
	'Pineapple': 'Dứa',
	'Sugerapple': 'Na',
	'Watermelon': 'Dưa hấu'
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fruit Detection using YOLOv11 and Webcam')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to the YOLOv11 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Webcam device number (default: 0)')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS on frame')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--lang', type=str, default='vi', choices=['en', 'vi'],
                        help='Language for display (en: English, vi: Vietnamese)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the YOLOv11 model
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the latest version of ultralytics installed.")
        print("Try: pip install -U ultralytics")
        return

    # Initialize webcam
    try:
        device_id = int(args.device)
    except ValueError:
        device_id = args.device  # Use as string (for example, video file path)

    print(f"Opening webcam device: {device_id}")
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your webcam connection or device ID.")
        return

    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Webcam resolution: {frame_width}x{frame_height}, FPS: {fps}")

    # Initialize video writer if save_video is True
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output video to {args.output}")

    # Create window
    cv2.namedWindow("Fruit Detection", cv2.WINDOW_NORMAL)

    print("Starting detection. Press 'q' to quit.")

    # Main loop
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # Start time for FPS calculation
        start_time = time.time()

        # Perform detection
        results = model.predict(frame, conf=args.conf, verbose=False)

        # Process results
        result = results[0]

        if args.lang == 'vi':
            # Tạo một bản sao của từ điển names để không ảnh hưởng đến kết quả gốc
            vi_names = result.names.copy()

            # Thay đổi tên lớp sang tiếng Việt trong bản sao
            for class_id, class_name in vi_names.items():
                if class_name in FRUIT_NAMES_VI:
                    vi_names[class_id] = FRUIT_NAMES_VI[class_name]

            # Lưu trữ tên gốc
            original_names = result.names

            # Tạm thời thay đổi tên lớp sang tiếng Việt
            result.names = vi_names

            # Vẽ kết quả với tên tiếng Việt
            annotated_frame = result.plot()

            # Khôi phục tên gốc
            result.names = original_names
        else:
            # Sử dụng tên tiếng Anh mặc định
            annotated_frame = result.plot()

        # Calculate FPS
        fps_value = 1.0 / (time.time() - start_time)

        # Display FPS on frame if requested
        if args.show_fps:
            cv2.putText(
                annotated_frame,
                f"FPS: {fps_value:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Show frame
        cv2.imshow("Fruit Detection", annotated_frame)

        # Save frame to video if requested
        if args.save_video and video_writer is not None:
            video_writer.write(annotated_frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
