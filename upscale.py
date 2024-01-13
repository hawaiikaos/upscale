from os import walk
from super_image import EdsrModel, ImageLoader
from PIL import Image

source_path = "./source/"
filenames = next(walk(source_path), (None, None, []))[2]  # [] if no file

print(filenames)

for f in filenames:
    fname = source_path + f
    print(fname)
    image = Image.open(r'{0}'.format(fname))
    # image = Image.open(requests.get(url, stream=True).raw)

    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(preds, './output/scaled_2x.png')
    ImageLoader.save_compare(inputs, preds, './output/scaled_2x_compare.png')