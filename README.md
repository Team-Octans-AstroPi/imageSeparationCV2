# imageSeparationCV2
OpenCV code that separates clouds and vegetation from an IR image of Earth. It also computes NDVI, which is used to extract vegetation & plant health information.

## Plant health
Using [our custom 8-bit colormap inspired from FastieCM](https://github.com/Team-Octans-AstroPi/imageSeparationCV2/blob/main/octanscm.py),
we split the plant health into 8 categories. Each category is assigned a specific color of the colormap. Then, for every category,
we compute how much of the vegetation in the image is in that category. Along with the vegetation percentage (how much of the image represents vegetation),
this information is stored inside the CSV used to train the climate model.

## Usage
- `main.py` is the file. You may create a Python venv and install its dependencies placed inside `requirements.txt`
- To generate the climate model's CSV file, check out this repository, which uses a modified version of this code.


