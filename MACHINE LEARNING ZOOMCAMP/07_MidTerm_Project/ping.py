from flask import Flask

app = Flask('ping') # give an identity to your web service

@app.route('/ping', methods=['GET']) # use decorator to add Flask's functionality to our function
def ping():
    return 'PONG'

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696



