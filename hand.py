import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks. python import vision

import cv2

import utilities as util


model_path = './models/hand_landmarker.task'


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# recent Data class is just a nice way of storing the most recent face detection
# data since it runs asyncynously
class RecentData:
  def __init__(self):
    self.results = None
    self.time = None

  def top(self):
    return self.results, self.time
  
  def pop(self):
    if self.results:
      holdRes = self.results
      holdTime = self.time

      self.results = None
      self.time = None

      return holdRes, holdTime
    else:
      return False, False
  
  def push(self,result, time):
    self.results = result
    self.time = time

# instantiate the data holder
dataHold = RecentData()


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

def process_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('hand landmarker result: {}'.format(result))
    #print(result.handedness)
    if result.handedness:
      #dataHold.pop()
      dataHold.push(result, timestamp_ms)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
    num_hands=2
    )
with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  #grab the cam to be used
  cap = cv2.VideoCapture(0)

  # Set the desired width and height for the capture
  # Try higher resolutions like 1280x720 or 1920x1080 for a wider field of view
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  # Use OpenCV’s VideoCapture to start capturing from the webcam.
  while cap.isOpened():

  # Create a loop to read the latest frame from the camera using VideoCapture#read()
    ret, frame = cap.read()

    if not ret:
      continue

    #quit program via "q" press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Send live image data to perform face detection.
    # The results are accessible via the `result_callback` provided in
    # the `FaceDetectorOptions` object.
    # The face detector must be created with the live stream mode.
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # Get current time in milliseconds

    landmarker.detect_async(mp_image, frame_timestamp_ms)

    res, ms = dataHold.top()
    #if res:
    #  print( abs(ms-frame_timestamp_ms))
    if res and 500 > abs(ms-frame_timestamp_ms):
      frame = util.draw_landmarks_on_image(frame,res)

    cv2.imshow('cam',frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))




cap.release()
cv2.destroyAllWindows()