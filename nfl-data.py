import cv2
import os
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import re
import pandas as pd




def compare_times(teams, img_time, image_path):
    df = pd.read_csv("pbp-2024.csv")
    time_match = re.search(r'(\d+):(\d+)', img_time).group()
    for i, row in df.iterrows():
        if row['OffenseTeam'] in teams and row['DefenseTeam'] in teams:
            quarter = str(row['Quarter'])
            time = str(row['Minute']) + ":" + str(row['Second'])
            if time == time_match and quarter == img_time[0]:
                print(f"Quarter {quarter}, {time} - {row['OffenseTeam']} vs {row['DefenseTeam']}")
                
                rng = list(range(int(image_path[26:30])-3, int(image_path[26:30])+1))
                im_list = []
                
                for num in rng:
                    fname = f"../extracted_frames/frame_{num:04d}.jpg"
                    image = cv2.imread(fname)
                    print(type(image))
                    im_list.append(image)

                data.loc[len(data)] = [im_list, quarter, time, row['Down'], row['ToGo'], row['PlayType']]


    # create a dictionary mapping play time to a list of every other value for each row
    # data:
    # x = [image, quarter, time, down, distance]
    # y = play_type



def extract_frames(video_path, output_folder, frame_interval=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    
    while frame_count < 10000:
        ret, frame = cap.read()
        if not ret:
            print("End")
            break
            
        # Save every nth frame (based on frame_interval)
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames")

# Usage
# extract_frames("BillsLionsWk15.mp4", "extracted_frames", frame_interval=20)



def search_word_in_image(image_path, search_word):
    # Load image with OpenCV
    image = cv2.imread(image_path)
    
    # Optional: convert to grayscale and threshold to improve OCR accuracy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    im = Image.open(image_path)
    print(im.size)

    cropped = im.crop((1050, 900, 1350, 1000))
    c_img = cv2.cvtColor(np.array(cropped), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(c_img, 150, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # Debug: show the cropped image
    # plt.imshow(cropped)
    # plt.axis('off')
    # plt.show()

    # OCR: extract text
    text = pytesseract.image_to_string(np.array(cropped))
    # if not text:
    #     plt.imshow(thresh, cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    #     text = pytesseract.image_to_string(thresh)

    # Debug: print full OCR text
   # print("Extracted Text:\n", text)

    # Check if the word exists in the text
    # if search_word.lower() in text.lower():
    #     print(f"Found the word '{search_word}' in the image.")
    #     return True
    # else:
    #     print(f"Did not find the word '{search_word}' in the image.")
    #     return False
    return text

# Example usage
image_path = "../extracted_frames/frame_0334.jpg"  # replace with your image path
text = search_word_in_image(image_path, "1st")


data = pd.DataFrame(columns=['image', 'quarter', 'time', 'down', 'distance', 'play_type'])

for i in range(20):
    image_path = f"../extracted_frames/frame_{i:04d}.jpg"
    text = search_word_in_image(image_path, "1st")
    if text:
        new = text
        if text[0] == "I":
            new = text.replace("I", "1")
        compare_times(["BUF", "DET"], new, image_path)

print(data.head())