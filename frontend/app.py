from flask import Flask, send_from_directory, request, Response
import requests

# Flask app serves files from current directory
app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def login_page():
    return send_from_directory('.', 'login.html')


# Proxy API requests to the backend FastAPI server to avoid 404s when users
# visit /api/* on the frontend host. This supports GET and POST (enough for
# the frontend usage in this project). If you prefer, you can remove this
# proxy and have the frontend call the backend directly at http://127.0.0.1:8000
@app.route('/api/<path:path>', methods=['GET', 'POST'])
def proxy_api(path):
    backend_url = f'http://127.0.0.1:8000/api/{path}'
    try:
        if request.method == 'GET':
            resp = requests.get(backend_url, params=request.args, timeout=5)
        else:
            resp = requests.post(backend_url, json=request.get_json(force=False, silent=True), timeout=5)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type'))
    except requests.exceptions.RequestException as e:
        return Response(f'Backend proxy error: {e}', status=502)


@app.route('/<path:path>')
def serve_page(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    # Run on port 5000 to avoid conflict with backend
    app.run(debug=True, port=5000)
