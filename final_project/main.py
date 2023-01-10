from flask import Flask

app = Flask(__name__)


def ping_view():
    return 'Hello, World!'


def query_view():
    pass


def update_index_view():
    pass


def create_app():

    # init code

    app.add_url_rule('/ping', view_func=ping_view)
    app.add_url_rule('/query', view_func=query_view)
    app.add_url_rule('/update_index', view_func=update_index_view)
    return app
