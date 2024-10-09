from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from main import ice_break_with

load_dotenv()

template_dir = os.path.abspath("../templates")
app = Flask(__name__, template_folder=template_dir)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, profile_pic_url = ice_break_with(name=name)
    return jsonify(
        {
            "summary_and_facts": summary.to_dict(),
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
