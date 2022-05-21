import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# UPLOAD_FOLDER = '../upload'
# ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        # f.save(f.filename)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'save_files',secure_filename(f.filename))
        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.abspath(upload_path) # 将路径转换为绝对路径
        f.save(upload_path)
        return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)