# app.py
import os
from flask import Flask, render_template, request
from prediction import predict_new_data, get_dropdown_options

app = Flask(__name__, template_folder="templates", static_folder="static")

# load dropdown options (this will try to read saved dropdowns from models/, or fallback to the xlsx)
dropdown_options = get_dropdown_options()

@app.route("/")
def index():
    return render_template("index.html", dropdown_options=dropdown_options)

@app.route("/predict", methods=["POST"])
def predict():
    # collect form fields (use 'Unknown' as default so prediction code can map it)
    form_data = {
        "classification": request.form.get("classification") or "Unknown",
        "code": request.form.get("code") or "Unknown",
        "implanted": request.form.get("implanted") or "Unknown",
        "name_device": request.form.get("name_device") or "Unknown",
        "name_manufacturer": request.form.get("name_manufacturer") or "Unknown",
    }

    predicted_class, description, suggestion = predict_new_data(form_data)

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        description=description,
        suggestion=suggestion,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_flag = os.environ.get("FLASK_DEBUG", "0") == "1"
    # Bind to 0.0.0.0 so deployment services (Render, Railway) can route to it
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
