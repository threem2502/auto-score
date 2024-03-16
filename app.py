from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import os
from library import process_grading

app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Kiểm tra xem có tệp ảnh và tệp kết quả được tải lên không
        if 'img_file' not in request.files or 'result_file' not in request.files:
            return jsonify({'error': 'Vui lòng tải lên cả ảnh và tệp kết quả.'})

        img_file = request.files['img_file']
        result_file = request.files['result_file']

        if img_file.filename == '' or result_file.filename == '':
            return jsonify({'error': 'Vui lòng chọn tệp hợp lệ.'})

        # Lưu tệp ảnh và tệp kết quả vào thư mục tạm thời
        img_path = 'temp_img.png'
        result_path = 'temp_result.xlsx'
        img_file.save(img_path)
        result_file.save(result_path)

        # Gọi hàm xử lý chấm điểm
        sbd,ma_de,score, inc = process_grading(img_path, result_path)

        # Xoá tệp ảnh và kết quả tạm thời sau khi xử lý xong
        os.remove(img_path)
        os.remove(result_path)

        return jsonify({'score': score,
                        'sbd': sbd,
                        'ma_de': ma_de,
                        'cau_sai': inc})
    except Exception as e:
        return jsonify({'error': 'Đã xảy ra lỗi: ' + str(e)})

if __name__ == '__main__':
    app.run(debug=True)
