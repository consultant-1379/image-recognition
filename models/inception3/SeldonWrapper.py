import predict_imagenet as p
import io
import base64

from PIL import Image

class SeldonWrapper(object):
    def __init__(self):
       self.loaded = False

    def class_names(self):
        return ["class","description","probability"]
    
    def load(self):
        print("Load model - start")
        self.model = p.load_model()
        self.model._make_predict_function()
        self.loaded = True
        print("Load model - end")

    def predict(self, X, features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called")

        if not self.loaded:
            self.load()

        base64_encoded_image = X[0]

        img_array = base64.b64decode(base64_encoded_image)
        pil_img = Image.open(io.BytesIO(img_array))

        prediction = p.predict(self.model, pil_img)

        print("Predict result: " + str(prediction))

        json_safe_prediction = [(p[0], p[1], p[2].item()) for p in prediction]

        return json_safe_prediction
