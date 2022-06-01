from flask import Flask,jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from image_class_api import crop
import tempfile
import urllib.request
import numpy as np
import base64
app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

ALLOWED_EXTENSIONS = set(['txt', 'pdf','jpg','jpeg', 'png', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Image(Resource):

    def post(self):
        errors = {}
        success = False

        if 'file' not in parser.parse_args():
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        args = parser.parse_args()
        the_file = args['file']
    
        if the_file and allowed_file(the_file.filename):
            #args = parser.parse_args()
            #the_file = args['file']
        
            # save a temporary copy of the file
            ofile, ofname = tempfile.mkstemp()
            the_file.save(ofname)
            # predict
            results = crop(ofname)
    
            success = True
        else:
            errors[the_file.filename] = 'File type is not allowed'
 
        if success and errors:
            errors['message'] = 'File(s) successfully uploaded'
            resp = jsonify(errors)
            resp.status_code = 500
            return resp
        if success:
            resp = jsonify({'message' : 'Files successfully uploaded'})
            resp.status_code = 201
            return results
        else:
            resp = jsonify(errors)
            resp.status_code = 500
            return resp




        
        
        


api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run(debug=True)