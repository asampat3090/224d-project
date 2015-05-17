#!/usr/bin/env python

import os

from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/tmp/captionly_demos_uploads'

@app.route('/')
def hello_world():
        return """
                <FORM METHOD='POST' ENCTYPE='multipart/form-data' ACTION='/upload'>
                File to upload: <INPUT TYPE=file NAME=attachment>
                <br>
                <INPUT TYPE=submit VALUE=Upload>
                </form>
        """

@app.route('/upload', methods=['POST'])
def upload():
        upload_file = request.files['attachment']

        # secure filename
        filename = secure_filename(upload_file.filename)

        # join to path
        filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # expand homedir (~)
        filename = os.path.expanduser(filename)

        upload_file.save(filename)

        return '''Success! File "%s" has been saved in %s''' % (filename, app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
        app.debug = True
        app.run(host='0.0.0.0', port=8080)