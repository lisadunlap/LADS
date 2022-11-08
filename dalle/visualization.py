from dalle2_laion import ModelLoadConfig, DalleModelManager
from dalle2_laion.scripts import InferenceScript
import torch
import numpy as np
from PIL import Image

class ExampleInference(InferenceScript):
    def run(self, text, image_embedding) -> Image:
        image_map = self._sample_decoder(text=text, image_embed=image_embedding)
        return image_map[0][0]

model_config = ModelLoadConfig.from_json_path("./dalle/dalle2.json")
model_manager = DalleModelManager(model_config)
inference = ExampleInference(model_manager)

original_embed = []
augmented_embed = []
with open('./dalle/embeddings.npy', 'rb') as f:
    for _ in range(10):
        a = np.load(f)
        original_embed.append(torch.from_numpy(a))
        b = np.load(f)
        augmented_embed.append(torch.from_numpy(b))

# original_domain = [1, 1, 0, 0, 0, 0, 1, 1, 0, 1]
# original_text = ['a photo of a landbird in the forest.', 'a photo of a waterbird on water.']
# augmented_text = ['a photo of a waterbird in the forest.', 'a photo of a landbird on water.']
for i in range(10):
    image = inference.run(["a photo"], [original_embed[i]])
    image.save(f'./dalle/images/waterbird-original-photo-{i}.png')
    image = inference.run(["a photo"], [augmented_embed[i]])
    image.save(f'./dalle/images/waterbird-augment-photo-{i}.png')

for i in range(10):
    image = inference.run(["a car"], [original_embed[i]])
    image.save(f'./dalle/images/waterbird-original-car-{i}.png')
    image = inference.run(["a car"], [augmented_embed[i]])
    image.save(f'./dalle/images/waterbird-augment-car-{i}.png')