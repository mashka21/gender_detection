#Follow all steps to get gender detection 

STEP-1  gender_detection

    #gender_detection
    !git clone https://github.com/mashka21/gender_detection.git
    %cd gender_detection

STEP-2  Downloading pretrained data and unzipping it

    # Downloading pretrained data and unzipping it
    !gdown https://drive.google.com/uc?id=12_ne5fObIMBxBFLSxsUu5l4M7tBar7Us
    # https://drive.google.com/uc?id=12_ne5fObIMBxBFLSxsUu5l4M7tBar7Us
    !unzip gender.zip

STEP-3  Import required modules

    # Import required modules
    import cv2 as cv
    import math
    import time
    from google.colab.patches import cv2_imshow

    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

    faceProto = "modelNweight/opencv_face_detector.pbtxt"
    faceModel = "modelNweight/opencv_face_detector_uint8.pb"

    genderProto = "modelNweight/gender_deploy.prototxt"
    genderModel = "modelNweight/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male', 'Female']

    # Load network
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    padding = 20

    def gender_detector(frame):
        # Read frame
        t = time.time()
        frameFace, bboxes = getFaceBox(faceNet, frame)
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            label = "{}".format(gender)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        return frameFace
    
    
STEP-4  image show / print the output 

    #To print the out of the image gender detection
    input = cv.imread("fitah.jpg")
    output = gender_detector(input)
    cv2_imshow(output)
