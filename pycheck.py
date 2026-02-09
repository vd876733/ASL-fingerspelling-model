from tensorflow.keras.models import load_model

try:
    model = load_model("static/Model/Keras_model.h5")
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print("❌ Error loading model:", e)
