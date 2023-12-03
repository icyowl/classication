import cv2
import pickle


with open('cl/data.pickle', 'rb') as f:
    data = pickle.load(f)
    print(data[0][0].shape, len(data[1]))
    # print(data[0][0])
    a = data[0][1]
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', a.reshape((28, 28)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()