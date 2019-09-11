from PIL import Image
from Networks.CustomNet import YOLOV3

if __name__ == '__main__':
    # Create Model
    yolov3 = YOLOV3((416, 416), load_model=2)
    while True:
        img = input('Input image filename:')
        if img == 'q':
            break
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolov3.detect_image(image)
            r_image.show()
    yolov3.close_session()
