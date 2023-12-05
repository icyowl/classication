import cv2
import pickle
from PIL import Image
import numpy as np
import os
import pytesseract  # type: ignore
from tqdm import tqdm
from jug.entry import pretty_printed_json


def get_downloads_path() -> str:  # abs path
    return os.path.join(os.path.expanduser("~"), 'Downloads')

def get_image_paths(path: str) -> list[str]:
    fnames = [fn for fn in os.listdir(path) if fn.endswith(('.jpeg', '.png'))]
    return [os.path.join(path, fn) for fn in fnames]

# --- image processing ---

def read_as_gray(path: str) -> np.ndarray:
    # cv2はマルチバイトのファイル名が読めない
    a = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)

def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    block_size = 11  # 領域のサイズ
    c = 2  # 平均あるいは加重平均から引かれる値
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    # return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)

def _masking(img: np.ndarray, x: int, y: int, w: int, h: int):
    img[y:y+h, x:x+w] = np.full((h, w), 0, dtype=np.uint8)
    return img

def masking(img: np.ndarray):
    h, w = img.shape
    mask = _masking(img, x=0, y=0, w=w, h=240) # 上部
    mask = _masking(mask, x=0, y=h-120, w=w, h=120)  # 下部
    mask = _masking(mask, x=0, y=0, w=30, h=h)  # 左端
    mask = _masking(mask, x=370, y=0, w=180, h=h)  # 中央（ART, 1/225)
    mask = _masking(mask, x=680, y=0, w=w-680, h=h)  # 右端
    return mask

def find_contours(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    # cv2.RETR_EXTERNALで外側の輪郭だけ取得する
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 17 or h > 25 or w > 25:  # target h x w -> 26 x 26
            continue
        selected.append((x, y, w, h))
    return selected

def padding_(img: np.ndarray, size=28):
    height, width = img.shape
    q, mod = divmod((size-height), 2)
    y = q + mod
    h = y + height
    q, mod = divmod((size-width), 2)
    x = q + mod
    w = x + width
    pad = np.full((size, size), 0, dtype=np.uint8)
    pad[y:h, x:w] = img
    return pad

def padding(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    p = 4
    pad = np.full((h+p*2, w+p*2), 0, dtype=np.uint8)
    pad[p:p+h, p:p+w] = img
    return pad

def find_contours_process(path: str) -> list[tuple[int, int, int, int]]:  # tuple[x, y, w, h]
    gray = read_as_gray(path)
    thres = adaptive_threshold(gray)
    mask = masking(thres)
    selected = find_contours(mask)
    return mask, selected

# --- ocr process ----------------------------------------

def get_selected_rows(selected: list) -> list:
    left_contours = [t for t in selected if t[0] < 60]
    sorted_left_contours = sorted(left_contours, key=lambda x:[x[1]])
    rows = []
    for left_t in sorted_left_contours:
        row = [t for t in selected if abs(left_t[1] - t[1]) < 60]
        sorted_row = sorted(row)
        rows.append(sorted_row)

    # len_rows = sum(len(row) for row in rows)
    # if len_rows != len(selected):
    #     print(f'Do not match. {len_rows} != {len(selected)}')

    return rows
    
def separate_into_chunks(row: list) -> list:
    chunks, que = [], []
    for i, contour in enumerate(row):
        x, y, w, h = contour
        if not i:
            pre_x = x + w
            que.append(contour)
        else:
            dst = x - pre_x
            # print(dst)
            if dst < 12:  # ここが10だと711が拾えない
                que.append(contour)
            else:
                q = que[:]
                chunks.append(q)
                que.clear()
                que.append(contour)
            pre_x = x + w
    chunks.append(que)
    return chunks

def ocr(char: Image.Image) -> str:
    # --oem 3 デフォルト。LSTMとTesseractエンジンを状況に応じて使用する
    # --psm 10 	画像を1文字とみなす
    config = r'-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    s = pytesseract.image_to_string(char, config=config)
    return s


# import pyocr
# import pyocr.builders

# def ocr(img: Image.Image) -> str:
#     tools = pyocr.get_available_tools()
#     tool = tools[0]
#     builder = pyocr.builders.DigitBuilder(tesseract_layout=10)
#     builder.tesseract_configs.append('-c')
#     builder.tesseract_configs.append('tessedit_char_whitelist="0123456789"')
#     text = tool.image_to_string(img, lang='eng', builder=builder)
#     return text

def ocr_process(path: str) -> dict:
    # main process
    arr, selected = find_contours_process(path)
    rows = get_selected_rows(selected)
    dic = {}
    for i, row in enumerate(rows):
        chunks = separate_into_chunks(row)
        text_data = []
        if len(chunks) == 4:
            for j, chunk in enumerate(chunks):
                text = ''
                for k, (x, y, w, h) in enumerate(chunk):
                    a = arr[y:y+h, x:x+w]
                    pa = padding(a)
                    pa_inv = cv2.bitwise_not(pa)
                    img = Image.fromarray(pa_inv)
                    if False:
                        num = '_'.join((str(i), str(j), str(k)))  # row _ chunk _ char
                        img.save('tests/images/img_' + num + '.jpg')
                    s = ocr(img)
                    text += s.strip()
                text_data.append(text)
            print(text_data, end=' ')
            s = text_data[0]
            if s == '100':
                s = s + '0'
            if s == '101':
                s = '1001'
            if len(s) > 2:
                key = int(s)
                values = [int(text) for text in text_data[1:]]
                dic[key] = values
                print(key, *values)
    return dic

def kuragano_numbers() -> tuple:
    im = [700] + list(range(721, 738)) + list(range(744, 761))
    my = list(range(681, 691)) + list(range(711, 721))
    go: list[int] = []
    return im, my, go

def kamisato_numbers() -> tuple:
    im = list(range(750, 776)) + list(range(787, 796)) + [1001]
    my = list(range(969, 977)) + list(range(993, 1001))
    go = list(range(776, 784))
    return im, my, go

# def yatojima_numbers():
#     im = list(range(561, 581)) + list(range(606, 611)) + list(range(621, 641))
#     my = list(range(581, 586)) + list(range(616, 621))
#     return im, my

def scan_oneday(date: str, hall='kuragano') -> None:
    dl = get_downloads_path()
    paths = get_image_paths(dl)

    dic: dict[int, int] = {}
    for path in paths:
        dic |= ocr_process(path)

    if hall == 'kuragano':
        im, my, go = kuragano_numbers()
        head = 'kg'
    if hall == 'kamisato':
        im, my, go = kamisato_numbers()
        head = 'ks'

    numbers = im + my + go
    for key in dic.keys():
        if key not in numbers:
            print(f'Invalid value: {key}')

    sign = '==' if len(numbers) == len(dic) else '!='
    print(f'hall: {len(numbers)} {sign} dic: {len(dic)}')

    for number in numbers:
        if number not in dic.keys():
            dic |= {number: [float('nan')] * 3}

    d: dict[str, dict] = {}
    d |= {'desc': {'hall': hall, 'date': date, 'sug': ''}}
    d |= {'imjug': {str(n): dic[n] for n in im}}
    d |= {'myjug': {str(n): dic[n] for n in my}}
    if go:
        d |= {'gojug': {str(n): dic[n] for n in go}}

    s = pretty_printed_json(d)
    with open('data/' + head + date + '.json', 'wt', encoding='utf-8') as f:
        f.write(s)


def like_mnist():
    X, t = [], []
    dl = get_downloads_path()
    paths = get_image_paths(dl)
    for path in paths:
        arr, selected = find_contours_process(path)
        rows = get_selected_rows(selected)
        for row in tqdm(rows):
            chunks = separate_into_chunks(row)
            if len(chunks) == 4:
                for chunk in chunks:
                    for x, y, w, h in chunk:
                        a = arr[y:y+h, x:x+w]
                        pa = padding(a)
                        pa_inv = cv2.bitwise_not(pa)
                        img = Image.fromarray(pa_inv)
                        s = ocr(img)
                        text = s.strip()
                        i = int(text) if text else float('NaN')
                        pa_ = padding_(a)
                        X.append(pa_.flatten())
                        t.append(i)

    with open('data.pickle', 'wb') as f:
        pickle.dump([X, t], f)

if __name__ == '__main__':
    # scan_oneday('20231202', hall='kamisato')
    like_mnist()

