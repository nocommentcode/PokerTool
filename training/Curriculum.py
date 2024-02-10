from training.Class import Class


from typing import List


class Curriculum:
    def __init__(self, classes: List[Class]):
        self.classes = classes

    def run(self, model, writer, device):
        total_epochs = 0
        for curriculum_class in self.classes:
            print(
                f"Starting class {curriculum_class.name} for {curriculum_class.epochs} epochs")

            curriculum_class.freeze_model_layers(model)
            curriculum_class.set_epoch_offset(total_epochs)

            curriculum_class.start(model, writer, device)

            total_epochs += curriculum_class.epochs
