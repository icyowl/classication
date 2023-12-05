import pickle
from cl.scan import *


def create_dataset():
    X, t = [], []
    dl = get_downloads_path()
    paths = get_image_paths(dl)
    for path in paths:
        arr, selected = find_contours_process(path)
        rows = get_selected_rows(selected)
        for row in tqdm(rows):
            chunks = separate_into_chunks(row)
            if len(chunks) == 4:
                for i, chunk in enumerate(chunks):
                    if i != 1:
                        for x, y, w, h in chunk:
                            a = arr[y:y+h, x:x+w]
                            pa = padding(a)
                            pa_inv = cv2.bitwise_not(pa)
                            img = Image.fromarray(pa_inv)
                            s = ocr(img)
                            text = s.strip()
                            value = int(text) if text else -1
                            pa_ = padding_(a)
                            X.append(pa_.flatten())
                            t.append(value)

    with open('data/data3.pickle', 'wb') as f:
        pickle.dump([X, t], f)


def show(img):
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mymnist_view():
    with open('data/data_a.pickle', 'rb') as f:
        data = pickle.load(f)
        print(len(data[0]))  # (784,) 839
    width = 30
    X, y = [], []
    for num in range(-1, 10):
        X = [X for X, y in zip(*data) if y == num]
        y = [(i, y) for i, (X, y) in enumerate(zip(*data)) if y == num]
        # if num == -1:
        #     l = []
        #     indices = 40, 41, 43, 45, 47, 49, 50, 52, 54, 55, 57, 59, 60, 62
        #     for idx in indices:
        #         print(y[idx])  # 367, 377, 388, 398, 408, 418, 428, 439, 450, 461
        #         # show(X[idx].reshape((28, 28)))
        #         l.append(y[idx][0])
    
        chars = [a.reshape((28, 28)) for a in X]
        remain = len(chars) % width
        chars = chars + [np.zeros((28, 28))] * (width - remain)
        hstacks = [np.hstack(chars[i:i+width]) for i in range(0, len(chars), width)]
        img = np.vstack(hstacks)
        show(img)

def fix_mymnist():
    with open('data/data3.pickle', 'rb') as f:
        data = pickle.load(f)
        X, y = data
        for i in [399, 409, 419, 428, 439, 449, 460, 470, 481, 491]:
            y[i] = 0
        for i in [401, 441, 462, 483]:
            y[i] = 8
    with open('data/data3a.pickle', 'wb') as f:
        pickle.dump([X, y], f)


def concat_mymnist():
    X, y = [], []
    for pkl in ['data1a.pickle', 'data2a.pickle', 'data3a.pickle',]:
        with open('data/' + pkl, 'rb') as f:
            data = pickle.load(f)
        X += data[0]
        y += data[1]
    with open('data/data_a.pickle', 'wb') as f:
        pickle.dump([X, y], f)

def predict():
    with open('data/clf_lr.pickle', 'rb') as f:
        clf = pickle.load(f)

    with open('data/data_a.pickle', 'rb') as f:
        data = pickle.load(f)
        X, y = data
    a = X[0]
    print(clf.predict([a]))

if __name__ == '__main__':
    # create_dataset()
    # mymnist_view()
    # fix_mymnist()
    # concat_mymnist()
    predict()