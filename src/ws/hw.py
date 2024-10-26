from flask import Flask

app = Flask('hello')

@app.route('/hello', methods=['GET'])
def ping():
    return "Welcome to the <b>real</b> world.."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

