from joblib import Memory

from flask import Flask, render_template, request, jsonify

from src.predict import Predict

memory = Memory('./tmp/cache', verbose=1)
app = Flask(__name__)
SEARCH_SIZE = 10

pred = Predict()


@memory.cache
def query(q, search_size=SEARCH_SIZE):
    return pred.search(q, n_answers=search_size)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    q = request.args.get('q')
    res = query(q)
    for i in range(len(res)):
        res[i]['start'] = int(res[i]['start'])
        res[i]['end'] = int(res[i]['end'])
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8300')
