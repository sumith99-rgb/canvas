"""
Camera diagnostic — tries every possible approach.
"""
import cv2
import time

print("=" * 50)
print("  Camera Diagnostic Tool")
print("=" * 50)

backends = [
    ("Default (no backend)", None),
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
]

for name, backend in backends:
    print(f"\nTrying {name}...")
    try:
        if backend is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0, backend)
        
        opened = cap.isOpened()
        print(f"  Opened: {opened}")
        
        if opened:
            # Try reading several frames
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    avg = frame.mean()
                    print(f"  Frame {attempt}: {w}x{h}, avg_brightness={avg:.1f}")
                    
                    # Save one frame for inspection
                    cv2.imwrite(f"test_frame_{name.split()[0].lower()}.jpg", frame)
                    print(f"  Saved test frame as test_frame_{name.split()[0].lower()}.jpg")
                    
                    # Show it
                    cv2.imshow(f"Camera Test - {name} - Press Q", frame)
                    print("  Showing frame... Press Q in the window to continue.")
                    
                    # Wait up to 5 seconds for user to see
                    start = time.time()
                    while time.time() - start < 8:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            break
                    cv2.destroyAllWindows()
                    break
                else:
                    print(f"  Frame {attempt}: failed to read")
                time.sleep(0.1)
        
        cap.release()
        time.sleep(1)  # Give camera time to release
        
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone.")
