from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from chatbotagent_test_copy import chatbot

# app = Flask(__name__, static_url_path='/react-app/build')
app = Flask(__name__)
CORS(app)


# # Serve static files from the 'react-app/build' directory
# @app.route('/react-app/build/static/<path:path>', methods=["GET"])
# def serve_static(path):
#     print("static")
#     return send_from_directory('./react-app/build', path)

# # Serve the index.html for the root and all other routes
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     if path != "" and os.path.exists(f'./react-app/build/{path}'):
#         return send_file(f'./react-app/build/{path}')
#     else:
#         print("hello")
#         return send_file('./react-app/build/index.html')

# Handle POST request to get response from chatbot
@app.route("/get_response", methods=["POST"])
def get_response():
    query = request.json.get('query')
    last5Chats = request.json.get('last5Chats')
    return jsonify(chatbot.ask_question(query, last5Chats))

# Example route
@app.route("/check")
def home():
    return jsonify("Hello, Flask")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
