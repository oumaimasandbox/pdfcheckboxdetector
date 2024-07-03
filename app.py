import streamlit as st
import os
import cv2
import numpy as np
import fitz  # PyMuPDF

# Custom CSS for better aesthetics
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding-top: 50px;
        padding-left: 10px;
        padding-right: 10px;
    }
    .main .block-container {
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to process the image and detect checkboxes using contours
def detect_boxes_contour(image, line_min_width=15, threshold_value=150, min_size=500, max_size=5000, min_ratio=0.8, max_ratio=1.2, extent_threshold=0.5):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, threshold_value, 255, cv2.THRESH_BINARY)
    img_bin = ~img_bin

    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final = img_bin_h | img_bin_v

    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(img_bin_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area  # Solidity

        if min_size < area < max_size and min_ratio < aspect_ratio < max_ratio and extent > extent_threshold:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Function to process the image and detect checkboxes using connected components
def detect_boxes_connected_components(image, line_min_width=15, threshold_value=150, min_size=500, max_size=5000, min_ratio=0.8, max_ratio=1.2, extent_threshold=0.5):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, threshold_value, 255, cv2.THRESH_BINARY)
    img_bin = ~img_bin

    # Morphological operations to detect lines
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final = img_bin_h | img_bin_v

    # Dilation to merge nearby components
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    for x, y, w, h, area in stats[2:]:
        aspect_ratio = w / float(h)
        rect_area = w * h
        extent = float(area) / rect_area  # Solidity

        if min_size < area < max_size and min_ratio < aspect_ratio < max_ratio and extent > extent_threshold:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Streamlit App
st.title("Checkbox Detector")

# Upload options
uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
if uploaded_file is not None:
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    total_pages = pdf_document.page_count

    # Detection settings in the left sidebar
    with st.sidebar:
        st.header("Detection Settings")
        with st.expander("Threshold Settings"):
            threshold_value = st.slider("Threshold Value", 0, 255, 150, help="Threshold value is used to binarize the image. Higher values will result in fewer detected features, while lower values will detect more features.")
        with st.expander("Line Width Settings"):
            line_min_width = st.slider("Line Min Width", 1, 50, 15, help="Minimum width of lines to be detected. Increase this value to detect wider lines.")
        with st.expander("Size Settings"):
            min_size = st.slider("Min Size", 100, 5000, 500, help="Minimum size of detected boxes. Increase this value to ignore smaller boxes.")
            max_size = st.slider("Max Size", 500, 10000, 5000, help="Maximum size of detected boxes. Decrease this value to ignore larger boxes.")
        with st.expander("Aspect Ratio Settings"):
            min_ratio = st.slider("Min Ratio", 0.1, 2.0, 0.8, step=0.1, help="Minimum aspect ratio of detected boxes. Adjust this value to filter boxes that are too narrow or too wide.")
            max_ratio = st.slider("Max Ratio", 0.1, 2.0, 1.2, step=0.1, help="Maximum aspect ratio of detected boxes. Adjust this value to filter boxes that are too narrow or too wide.")
        with st.expander("Extent Settings"):
            extent_threshold = st.slider("Extent Threshold", 0.0, 1.0, 0.5, step=0.1, help="Solidity of detected boxes. Increase this value to detect more solid boxes and decrease to detect more hollow boxes.")
        detection_function = st.selectbox(
            "Detection Function", 
            ["Contour Detection", "Connected Components"], 
            help=(
                "Select the detection method. "
                "'Contour Detection' looks for continuous lines that form the outline of a shape. "
                "It is useful when the shapes you want to detect are well-defined and have clear edges. "
                "'Connected Components' groups together areas of the image that are connected or close to each other. "
                "It is useful for detecting regions of interest that are closely packed together or when the objects have a more filled or solid appearance."
            )
        )

    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("Page Navigation")
        page_number = st.slider("Page Number", 1, total_pages, 1)
        page = pdf_document.load_page(page_number - 1)
        zoom = 2.0  # Increase the resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:  # Check for alpha channel and remove it
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        st.image(img, caption=f'Page {page_number}', use_column_width=True)

    with col1:
        st.write("Debug Info: Image Shape - ", img.shape)
        image_copy = img.copy()
        if detection_function == "Contour Detection":
            output_image = detect_boxes_contour(image_copy, line_min_width=line_min_width, threshold_value=threshold_value, min_size=min_size, max_size=max_size, min_ratio=min_ratio, max_ratio=max_ratio, extent_threshold=extent_threshold)
        else:
            output_image = detect_boxes_connected_components(image_copy, line_min_width=line_min_width, threshold_value=threshold_value, min_size=min_size, max_size=max_size, min_ratio=min_ratio, max_ratio=max_ratio, extent_threshold=extent_threshold)

        st.image(output_image, caption='Detected Checkboxes', use_column_width=True)