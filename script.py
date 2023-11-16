import torchvision

from src.model_api.processor import Processor
import src.lung.constants as c
processor = Processor()
# processor.run_training(update_data=False)
# processor.run_test(False)

# prediction = processor.predict(os.path.join("./images/TorsoXray.png"))
prediction = processor.predict('/Users/jeschrader/code-port/simple-unet-2/segmentation/train/image/CHNCXR_0061_0.png')
torchvision.utils.save_image(
    prediction, f"saved_images/prediction.png"
)
