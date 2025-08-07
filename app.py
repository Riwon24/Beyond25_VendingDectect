uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    input_tensor = preprocess_image(img)
    output = run_inference(session, input_tensor)
    detected_items = extract_detected_items(output)
    result_img = draw_boxes(np.array(img), output)

    st.image(result_img, caption="탐지 결과")
    st.write("탐지된 품목:", detected_items)
