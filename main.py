from opencv import cv2
import numpy as np
import streamlit as st
from PIL import Image


def lemon(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2HLS)
    return img


def BlackBerry(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
    return img


def candy(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_RGB2LAB)
    return img


def mango(input_img1):
    img_gray = cv2.cvtColor(input_img1, cv2.COLOR_HSV2BGR)
    B, G, R = cv2.split(img_gray)
    zeros = np.zeros(input_img1.shape[:2], dtype="uint8")
    merge_img = cv2.merge([G, G, R])
    return merge_img


def roses(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2RGB)
    color = cv2.blur(img, (5, 5))
    return color


def dimond(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_Lab2LBGR)
    Glass = cv2.GaussianBlur(img, (5, 5), sigmaX=2)
    return Glass


def Edges(input_img1):
    img = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img)
    img_edge = cv2.GaussianBlur(img_invert, (21, 21), 0)
    img_candy = cv2.Canny(img_edge, 5, 5)
    ret, mask = cv2.threshold(img_candy, 70, 2515, cv2.THRESH_BINARY)
    return mask


st.title("Image filter App")

input_file = st.sidebar.file_uploader(
    "Upload Image", type=['jpg', 'png', 'jpeg'])

select_img = st.sidebar.selectbox(
    "select option", ("lemon", "Black Berry", "Candy", "Mongo", "Roses", "Dimond", "Sunflower", "Edge",))
if input_file is None:
    st.write("Enter Image")
else:
    st.image(input_file, width=400)

    input_img = Image.open(input_file)
    input_img1 = np.array(input_img)
    st.write("Output Image")
    if select_img == "lemon":
        img_lemon = lemon(input_img1)
        st.image(img_lemon, width=450)

    elif select_img == "Black Berry":
        Black = BlackBerry(input_img1)
        st.image(Black, width=450)

    elif select_img == "Candy":
        img_candy = candy(input_img1)
        st.image(img_candy, width=450)

    elif select_img == "Mongo":
        img_mango = mango(input_img1)
        st.image(img_mango, width=450)

    elif select_img == "Roses":
        img_smooth = roses(input_img1)
        st.image(img_smooth, width=450)
    elif select_img == "Dimond":
        img_dimond = dimond(input_img1)
        st.image(img_dimond, width=450)

    elif select_img == "Sunflower":
        img_gray = cv2.cvtColor(input_img1, cv2.COLOR_LAB2RGB)
        B, G, R = cv2.split(img_gray)
        zeros = np.zeros(input_img1.shape[:2], dtype="uint8")
        merge_img = cv2.merge([G, G, R])
        st.image(merge_img, width=450)

    elif select_img == "Edge":
        img_edge = Edges(input_img1)
        st.image(img_edge, width=450)
