mport subprocess
import cv2
import os

token = subprocess.run(["adb", "shell", "rtsp-token", "192.168.50.69"], stdout=subprocess.PIPE).stdout.decode()
uri = f"rtsp://192.168.50.69:8554/mr/1080/9_10/token={token}"
print(uri)


#os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(uri)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
 BELOW CODE IS TO GENERATE NEW TOKEN. DOESN'T WORK. 


token_path = r'C:\\Users\\keeth\\Desktop\\platform-tools\\livetoken.txt'
bat_path = r'C:\\Users\\keeth\\Desktop\\platform-tools\\savetoken.bat'
print(bat_path)
subprocess.call([bat_path])
with open(token_path, "r") as f:
    token = f.read()    
    print(token)
    f.close()
    print(f.closed)
    



RTSP_URL = 'rtsp://192.168.50.59:8554/mr/1080/9_10/token='+token