from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import PIL
import pickle
import numpy as np
import cv2
import keras
from keras.models import model_from_json
from tkinter import ROUND
import os

json_file = open("model_final.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_final.weights.h5")
# loaded_model = open("trained_model.p", "rb")
model = loaded_model

from tensorflow.keras.models import load_model

# model = load_model("best_model.h5")



lastx, lasty = None, None
is_uploaded_image = False 



def clear_widget():
    global draw_board, image1, draw, text, is_uploaded_image
    image1 = PIL.Image.new("RGB", (900, 400), (255, 255, 255))
    text.delete(1.0, END)
    draw = ImageDraw.Draw(image1)
    draw_board.delete('all')
    is_uploaded_image = False  
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    draw_board.create_line((lastx, lasty, x, y), width=4, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    draw.line([lastx, lasty, x, y], fill="black", width=1)
    lastx, lasty = x, y

def activate_event(event):
    global lastx, lasty, is_uploaded_image
    draw_board.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y
    is_uploaded_image = False 

def load_image():
    """Tải hình ảnh từ máy tính và hiển thị lên canvas"""
    global draw_board, image1, draw, is_uploaded_image
    
    
    file_path = filedialog.askopenfilename(
        title="Chọn hình ảnh",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
           
            loaded_img = Image.open(file_path)
            
           
            canvas_width, canvas_height = 900, 400
            img_width, img_height = loaded_img.size
            
           
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
           
            resized_img = loaded_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            
            canvas_img = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
            
            
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            
            canvas_img.paste(resized_img, (x_offset, y_offset))
            
            image1 = loaded_img
            image2 = canvas_img
            draw = ImageDraw.Draw(image2)
            is_uploaded_image = True  
           
            photo = ImageTk.PhotoImage(canvas_img)
            draw_board.delete('all')
            draw_board.create_image(canvas_width//2, canvas_height//2, image=photo)
            draw_board.image = photo  
            
           
            text.delete(1.0, END)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải hình ảnh: {str(e)}")


def merge_contours_morphology(image, contours):
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
   
    new_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return new_contours

def resize_with_padding(image, target_size=(45, 45), pad_color=0):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_w = (target_size[1] - new_w) // 2
    pad_h = (target_size[0] - new_h) // 2
    
    padded = cv2.copyMakeBorder(resized, pad_h, target_size[0] - new_h - pad_h,
                                pad_w, target_size[1] - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded
def process_uploaded_image():
   
    filename = 'image_out.png'
    image1.save(filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        
        thresh1 = cv2.adaptiveThreshold(
            img, 255,   
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=23,
            C=8
        )

        
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        
        mask = np.ones(thresh1.shape, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if area < 10:
                if aspect_ratio <= 2:
                    cv2.fillPoly(mask, [contour], 0)

        
        result = cv2.bitwise_and(thresh1, mask)
        dilation = cv2.dilate(result, kernel_dilate, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        cv2.imshow('a',closing)
        cv2.waitKey(0)
        
        
        thresh = cv2.ximgproc.thinning(closing)
       
            
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Lọc contour lần 2
        # mask = np.ones(thresh.shape, dtype=np.uint8) * 255
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     x, y, w, h = cv2.boundingRect(contour)
        #     aspect_ratio = w / h if h > 0 else 0
            
        #     if area < 1:
        #         if aspect_ratio <= 3:
        #             cv2.fillPoly(mask, [contour], 0)

        # result = cv2.bitwise_and(thresh, mask)
        cv2.imshow('as',thresh)
        cv2.waitKey(0)
        # Tìm contours và sắp xếp
        ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contours = merge_contours_morphology(thresh, ctrs)
        cnt = sorted(merged_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        rects = [cv2.boundingRect(c) for c in cnt]
        
        final_rect = []
        for r in rects:
            x, y, w, h = r
            ratio = w / h
            max_height = 8
            min_width = 12
            min_ratio = 2
            
            if ratio >= min_ratio and h <= max_height and w >= min_width:           
                final_rect.append(r)
            else:
                if w * h > 250:
                    final_rect.append(r)
        
       
        s = ''
        image_display = cv2.imread(filename)
        
        for r in final_rect:
            x, y, w, h = r
            cv2.rectangle(image_display, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            padding = 5
            im_crop = thresh[max(0, y-padding):min(y+h+padding, thresh.shape[0]),
                            max(0, x-padding):min(x+w+padding, thresh.shape[1])]
            
            
            im_resize = resize_with_padding(im_crop)
            cv2.imshow('a',im_resize)
            cv2.waitKey(0)
            im_resize = im_resize / 255.0
            
            im_resize = np.array(im_resize)
            im_resize = im_resize.reshape(1, 45, 45, 1)
            result_pred = model.predict(im_resize)[0]
            final_pred = np.argmax(result_pred, axis=-1)
            
           
            if final_pred == 11:
                label = "-"
                s += '-'
            elif final_pred == 10:
                label = "+"
                s += '+'
            elif final_pred == 12:
                label = "*"
                s += '*'
            elif final_pred == 13:
                label = "/"
                s += '/'
            elif final_pred == 14:
                label = "("
                s += '('
            elif final_pred == 15:
                label = ")"
                s += ')'
            else:
                label = str(final_pred)
                s += str(final_pred)
            
            
            data = label + ' ' + str(int(max(result_pred) * 100)) + '%'
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 0)
            thickness = 1
            cv2.putText(image_display, data, (x, y - 5), font, fontScale, color, thickness)
            
        
        print("Extracted string:", s)
        
       
        try:
            result_calc = eval(s)
            print("Kết quả:", result_calc)
            text.insert(END, s+ "=" + str(result_calc))
        except Exception as e:
            print("Lỗi tính toán:", e)
            text.insert(END, s)
            
        
    
        cv2.imshow('Processed Image', image_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_drawn_image():
    """Xử lý hình ảnh vẽ tay với thuật toán cũ"""
    filename = 'image_out.png'
    image1.save(filename)
    image = cv2.imread(filename)

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = gray 
    thresh1 = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=23,
            C=8
        )

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

    dilation = cv2.dilate(thresh1, kernel_dilate, iterations=1)
    
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    thresh = cv2.ximgproc.thinning(closing)
    
    
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged_contours = merge_contours_morphology(image, ctrs)
    
    cnt = sorted(merged_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    rects = [cv2.boundingRect(c) for c in cnt]

    final_rect = []
    for r in rects:
        x, y, w, h = r
        print(w,h)
        if w * h > 2: 
            final_rect.append(r)
            
    s = ''
    for r in final_rect:
        x, y, w, h = r
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
       
        padding = 4
        im_crop = thresh[max(0, y-padding):min(y+h+padding, thresh.shape[0]),
                        max(0, x-padding):min(x+w+padding, thresh.shape[1])]
        
        
        im_resize = resize_with_padding(im_crop)
        im_resize = im_resize / 255.0
        cv2.imshow('anh dua vao predict',im_resize)
        cv2.waitKey(0)
        im_resize = np.array(im_resize)
        im_resize = im_resize.reshape(1, 45, 45, 1)
        result = model.predict(im_resize)[0]
        final_pred = np.argmax(result, axis=-1)
        
        if final_pred == 11:
            label = "-"
            s += '-'
        elif final_pred == 10:
            label = "+"
            s += '+'
        elif final_pred == 12:
            label = "*"
            s += '*'
        elif final_pred == 13:
            label = "/"
            s += '/'
        elif final_pred == 14:
            label = "("
            s += '('
        elif final_pred == 15:
            label = ")"
            s += ')'
        else:
            label = str(final_pred)
            s += str(final_pred)
        
        data = label + ' ' + str(int(max(result) * 100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
    
    print("Extracted string:", s)
    
    try:
        result_calc = eval(s) 
        print("Kết quả:", result_calc)
        text.insert(END, s+ "=" + str(result_calc))
    except Exception as e:
        print("Lỗi tính toán:", e)
        text.insert(END, s)
    
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save():
    """Hàm chính để xử lý - phân biệt giữa hình tải lên và vẽ tay"""
    global is_uploaded_image
    
    if is_uploaded_image:
        print("Processing uploaded image...")
        process_uploaded_image()
    else:
        print("Processing drawn image...")
        process_drawn_image()


win = Tk()
win.geometry("950x750") 
win.title("Multiple Handwritten Digit Recognition")
win.config(background="#66c2ff")

fontStyle = tkFont.Font(family="Lucida Grande", size=15)

write_label = Label(win, text="Write your number or upload image:", bg="#66c2ff", font=fontStyle)
write_label.place(relx=0.03, rely=0.03)

draw_board = Canvas(win, width=900, height=400, bg='white')
draw_board.place(relx=0.03, rely=0.1)
draw_board.bind('<Button-1>', activate_event)

image1 = PIL.Image.new("RGB", (900, 400), (255, 255, 255))
draw = ImageDraw.Draw(image1)


upload_btn = Button(text="Upload Image", command=load_image, bg="#4CAF50", fg="white", 
                   font=tkFont.Font(family="Lucida Grande", size=15))
upload_btn.place(relx=0.15, rely=0.63, anchor=CENTER)

button = Button(text="Extract", command=save, bg="#66c2ff", 
               font=tkFont.Font(family="Lucida Grande", size=20))
button.place(relx=0.5, rely=0.63, anchor=CENTER)

del_btn = Button(win, text="Erase All", command=clear_widget, bg="#f44336", fg="white", 
                width=8, font=tkFont.Font(family="Lucida Grande", size=15))
del_btn.place(relx=0.85, rely=0.63, anchor=CENTER)

predict_label = Label(win, text="Extracted Number:", bg="#66c2ff", 
                     font=tkFont.Font(family="Lucida Grande", size=13))
predict_label.place(relx=0.03, rely=0.72)

text = Text(win, height=2, width=25, font=tkFont.Font(family="Lucida Grande", size=13))
text.place(relx=0.03, rely=0.79)

win.mainloop()