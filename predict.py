import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import glo_var
import model
import img_gen


def display(restored_model, image_path, label):
    img_arr = img_gen.Image2Tensor(image_path)
    # the batch contains only one image
    img_bc = np.expand_dims(img_arr, 0)
    
    predictions = restored_model.predict(img_bc)
    result = [np.argmax(predictions[i]) for i in range(len(predictions))]
    result = img_gen.Index2Chars(result)
    
    img = Image.open(image_path)
    plt.imshow(img)
    plt.xlabel("Expected: %s\n  Answer: %s" % (label, result))
    plt.show()


model_predict = model.Get_model()

if os.path.exists(os.path.join(glo_var.checkpoint_dir, "checkpoint")):
    model_predict.load_weights(glo_var.checkpoint_path)

    for _, _, filenames in os.walk(glo_var.TEST_DATA_DIR):
        for filename in filenames:
            label = filename.split('.')[0].split('_')[-1]
            display(model_predict, os.path.join(glo_var.TEST_DATA_DIR, filename), label)
    
else:
    print("Model Not Found")
