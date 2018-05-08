###################################################
# Start bottle server to receive files
import bottle
from bottle import run, request, post, app, route
import tempfile
import os

# Handle CORS requests
def cors(func):
    def wrapper(*args, **kwargs):
        bottle.response.set_header("Access-Control-Allow-Origin", "*")
        bottle.response.set_header("Access-Control-Allow-Origin", "GET, POST, OPTIONS")
        bottle.response.set_header("Access-Control-Allow-Headers", "Origin, Content-Type")
        
        # skip the function if it is not needed
        if bottle.request.method == 'OPTIONS':
            return

        return func(*args, **kwargs)
    return wrapper


#@route('/', methods='GET OPTIONS'.split())
#@cors

@post('/')
def index():
    postdata = request.body.read().decode("utf-8")
    #print(postdata)

    f = open(os.path.join(tempfile.gettempdir(), "dataFile"), "w")
    f.write(postdata)
    f.close()

run(host='localhost', port=15000, debug=True)

###################################################
