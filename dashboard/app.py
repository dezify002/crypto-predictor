import os

from flask import Flask, render_template, jsonify, request

from dashboard_utils import Dashboard

app = Flask(__name__)

dashboard = Dashboard()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(dashboard.load())


@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.json

    try:
        result = dashboard.predict(
            asset=data["asset"],
            target_price=data["target"],
            minutes=data["minutes"],
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )