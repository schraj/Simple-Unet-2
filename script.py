from src.processor import Processor
processor = Processor()
processor.run_training(update_data=False)

processor.run_test()
