from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route("/")
def home():
    content = ""
    if os.path.exists("../logs.txt"):
        with open("../logs.txt", "r") as f:
            content = f.read()

    html = f"""
    <h1>Dashboard del Proyecto</h1>
    <h2>Logs</h2>
    <pre>{content}</pre>
    """

    return render_template_string(html)


if __name__ == "__main__":
    app.run(port=5000, debug=False)
