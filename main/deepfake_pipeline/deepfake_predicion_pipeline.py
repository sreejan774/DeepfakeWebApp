import cv2
from facenet_pytorch import MTCNN
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import concatenate, Dense
from keras.models import Model

BASE_DIR = os.getcwd()
MODEL_FOLDER = os.path.join(BASE_DIR,"main/deepfake_pipeline/trained_models")
PROCESSED_OUTPUT_DIR = os.path.join(BASE_DIR,"main/deepfake_pipeline/processed_output")


def extract_face_for_single_video(video_path,output_path):
    
    mtcnn = MTCNN()
    
    video_name = video_path.split('/')[-1]
    print(f'processing {video_name}')
    face = 'face'
    flow = 'flow'
    real = 'real'
    fake = 'fake'
    os.mkdir(os.path.join(output_path,face))
    os.mkdir(os.path.join(output_path,flow))
    facePath = os.path.join(output_path,face)
    flowPath = os.path.join(output_path,flow)
    
    os.mkdir(os.path.join(flowPath,real))
    os.mkdir(os.path.join(flowPath,fake))
    
    print(f"Calculating Bounding Box for {video_name}")
    xmin = 100000
    ymin = 100000
    xmax = -100000
    ymax = -100000
    
    cap = cv2.VideoCapture(video_path)
    frameCount = 0
    while(cap.isOpened()):
        ret,frame = cap.read()
        
        if ret == True:
            result,probs = mtcnn.detect(frame,)
            try:
                x_min, y_min, x_max, y_max = result[0]
                if(x_min-30 >= 0):
                    x_min = x_min - 30
                if(y_min-30 >= 0):
                    y_min = y_min - 30 
                x_max = x_max + 60
                y_max = y_max + 60

                xmin = min(xmin,x_min)
                ymin = min(ymin,y_min)
                xmax = max(xmax,x_max)
                ymax = max(ymax,y_max)
                
                frameCount += 1
                
                if frameCount == 120:
                    break
            except:
                pass
        else:
            break

    cap.release()
    
    print(xmin,xmax,ymin,ymax)

    print(f"Cropping frame of {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    
    frameCount = 1
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
          try:        
              frame = frame[int(ymin):int(ymax), int(xmin):int(xmax),:]

              num = '_' + str(frameCount).zfill(4)
              frameCount += 1
              #print(num)
              frame_name = video_name + num +'.jpg'  
              #print(frame_name)
              destination = os.path.join(facePath,frame_name)

              cv2.imwrite(destination,frame)

              if cv2.waitKey(25) & 0xFF == ord('q'):
                  break
              if frameCount == 120:
                  break
          except:
                print("Error during cropping")
                pass
                
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return (f"Processed {video_name}",flowPath)

def extract_flow_for_single_video(video_frame_path, output_path, params=[], to_gray=False):
    # Read the first frame
    print(f"Processing {video_frame_path}")
    method = cv2.optflow.calcOpticalFlowDenseRLOF
    frame_list = [os.path.join(video_frame_path,file) for file in os.listdir(video_frame_path)]
    frame_list.sort()
    print(frame_list[0])
    old_frame = cv2.imread(frame_list[0])
    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    count = 1
    while True:
        # Read the next frame
        new_frame = cv2.imread(frame_list[count])
        frame_copy = new_frame
        count+=1
        if count == 119:
            break
            
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("frame", frame_copy)
        # cv2.imshow("optical flow", bgr)
        frame_name = frame_list[count-2].split('/')[-1]
        frame_path = os.path.join(output_path,frame_name)
        cv2.imwrite(frame_path,bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame         
    return f"Processed {video_frame_path}"

def preprocessing(input_video_path, output_path):
    if(len(os.listdir(output_path)) == 0):
        dir_name = "1"
        os.mkdir(os.path.join(output_path,dir_name))
    else:
        dirs = os.listdir(output_path)
        dir_name = str(len(dirs) + 1) 
        os.mkdir(os.path.join(output_path,dir_name))
    face = 'face'
    real = 'real'
    new_output_path = os.path.join(output_path,dir_name)
    result = extract_face_for_single_video(input_video_path, new_output_path)
    flow_output_path = os.path.join(result[1],real)
    extract_flow_for_single_video(os.path.join(new_output_path,face),flow_output_path)
    return os.path.join(os.getcwd(),result[1])

def predict_from_images(list_of_models, ensambled_model, test_img_dir_path):
    test_datagen = ImageDataGenerator(
        rescale = 1./255,    #rescale the tensor values to [0,1]
    )
    input_size = 150

    test_generator = test_datagen.flow_from_directory(
        directory = test_img_dir_path,
        classes=['real', 'fake'],
        target_size = (input_size, input_size),
        color_mode = "rgb",
        class_mode = None,
        batch_size = 1,
        shuffle = False
    )
    
    preds = []
    for model in list_of_models:
        yhat = model.predict(test_generator,verbose = 1)
        preds.append(yhat)
    testX = np.dstack((preds[0],preds[1],preds[2],preds[3]))
    testX = testX.reshape((testX.shape[0], testX.shape[1]*testX.shape[2]))
    meta_predictions = ensambled_model.predict(testX)
    prediction = np.average(meta_predictions)
    return prediction

def predict(input_video_path,output_folder=PROCESSED_OUTPUT_DIR):
    print("BASE_DIR_PATH", BASE_DIR)
    modelPath1 = os.path.join(MODEL_FOLDER,"densenet201.h5")
    modelPath2 = os.path.join(MODEL_FOLDER,"inceptionV3.h5")
    modelPath3 = os.path.join(MODEL_FOLDER,"inceptionV3.h5")
    modelPath4 =  os.path.join(MODEL_FOLDER,"vgg19.h5")
    ensambledModel = os.path.join(MODEL_FOLDER,"ensambled_model.h5")

    model1 = load_model(modelPath1)
    model2 = load_model(modelPath2)    # todo deal with this shit
    model3 = load_model(modelPath3)
    model4 = load_model(modelPath4)
    model  = load_model(ensambledModel)

    members = [model1, model2, model3, model4]
    test_img_dir_path = preprocessing(input_video_path,output_folder)
    return predict_from_images(members,model,test_img_dir_path)