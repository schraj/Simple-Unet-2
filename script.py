import torchvision
import os

from src.model_api.processor import Processor
processor = Processor()
# processor.run_training(update_data=False)

prediction = processor.predict(os.path.join("./images/TorsoXray.png"))
torchvision.utils.save_image(
    prediction, f"saved_images/prediction.png"
)

