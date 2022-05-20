from flask import Flask, render_template, request


# UPLOAD_FOLDER = '../tmp'
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
        f.save(f.filename)
        return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)