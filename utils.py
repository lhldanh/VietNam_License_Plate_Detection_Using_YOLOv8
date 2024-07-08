import cv2
import easyocr


def process_result(result):

    int_to_char = {
        "6": "G",
        "0": "O",
        "5": "S",
        "4": "A"
    }
    char_to_int = {
        "g": "9",
        "L": "4",
        "Q": "0",
        "I": "1",
        "G": "6",
        "O": "0",
        "e": "8"
    }

    if len(result) == 1:
        text = list(result[0])
        for i in range(0, 2):
            if (text[i] in char_to_int.keys()):
                text[i] = char_to_int[text[i]]
        if (text[2] in int_to_char.keys()):
            text[2] = int_to_char[text[2]]
        for i in range(1, 6):
            if (text[-i] in char_to_int.keys()):
                text[-i] = char_to_int[text[-i]]
        result[0] = ''.join(text)
    if len(result) == 2:
        text = list(result[0])
        for i in range(0, 2):
            if (text[i] in char_to_int.keys()):
                text[i] = char_to_int[text[i]]
        if (text[2] in int_to_char.keys()):
            text[2] = int_to_char[text[2]]
        result[0] = ''.join(text)
        text = list(result[1])
        for i in range(0, len(text)):
            if (text[i] in char_to_int.keys()):
                text[i] = char_to_int[text[i]]
        result[1] = ''.join(text)

    if len(result) == 3:
        text = list(result[0])
        for i in range(0, 2):
            if (text[i] in char_to_int.keys()):
                text[i] = char_to_int[text[i]]
        text = list(result[2])
        for i in range(0, len(text)):
            if (text[i] in char_to_int.keys()):
                text[i] = char_to_int[text[i]]
        result[2] = ''.join(text)
    result = "-".join([res for res in result])
    result = result.replace("I", "1")
    return result


def extract_plate_text_easy_ocr(img, xmin, ymin, xmax, ymax):

    cropped_img = img[ymin:ymax, xmin:xmax]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_img)
    result = [res[1] for res in result]
    plate_text = process_result(result)

    return plate_text


def visualize_plate(
        img_path, predictions,
        conf_thres=0.7,
        font=cv2.FONT_HERSHEY_SIMPLEX
):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    for prediction in predictions:
        conf_score = prediction['confidence']

        if conf_score < conf_thres:
            continue

        bbox = prediction['box']
        xmin = int(bbox['x1'])
        ymin = int(bbox['y1'])
        xmax = int(bbox['x2'])
        ymax = int(bbox['y2'])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        text = extract_plate_text_easy_ocr(img, xmin, ymin, xmax, ymax)
        print(text)
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)

        cv2.rectangle(
            img,
            (xmin, ymin - text_height - 5),
            (xmin + text_width, ymin),
            (0, 255, 0),
            -1
        )
        cv2.putText(img, text, (xmin, ymin - 5), font, 1, (0, 0, 0), 2)

    return (img, text)
