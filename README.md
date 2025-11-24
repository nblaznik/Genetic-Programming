# Evolutionary Polygon Painter

This project reconstructs an image using **evolutionary algorithms** and **semi-transparent polygons**.  
A population of polygons (“DNA”) evolves toward the target image over many iterations.  
Each iteration mutates the DNA, renders it, evaluates similarity on the GPU, and keeps only improvements.

The result is a stylized, generative reconstruction of your input.

Gif Example: 
<p align="center">
  <img src="gifs/images_trophy_shapes_200_50000.gif" width="400">
</p>
---

## Features

- Evolve images using multi-point polygons with transparency  
- GPU-accelerated fitness evaluation (PyTorch)  
- Automatic polygon addition and removal  
- Ignores white backgrounds using a foreground mask  
- Edge-weighted fitness for better structural matching  
- DNA saved as JSON for full reproducibility  
- Exact reconstruction possible at any time  
- Optional export to SVG or Tailwind/CSS clip-path shapes  

---

## How It Works

Each polygon in the DNA contains:

- a list of vertex points  
- an RGBA color (with alpha)

This fully defines its:

- position  
- orientation  
- size  
- shape  
- transparency  

### Evolution Loop

1. **Initialize** with random polygons  
2. **Mutate** one polygon (change vertices or colors) or add/remove polygons  
3. **Render** the polygons onto a blank canvas  
4. **Evaluate** the difference to the target image  
5. **Accept** the mutation only if it improves the fitness  
6. **Repeat** thousands of times  

Intermediate steps and DNA states are saved inside the `images/` directory.

---

## Requirements

Install dependencies:

```
pip install pillow numpy opencv-python torch matplotlib
```

---

## Usage

Place your source image in the project root:

```
nejc_shapes.png
```

Run the evolutionary painter:

```
python evolve_gpu.py
```

Output files will appear in:

```
images/
  step_000500.png
  step_001000.png
  ...
  final.png
  final_dna.json
```

---

## Reconstructing the Image

The DNA file contains all geometry and color information needed to redraw the polygons.

Example:

```python
from PIL import Image
from dna_utils import load_dna, render_dna   # or your versions
import matplotlib.pyplot as plt

dna = load_dna("images/final_dna.json")
target = Image.open("nejc_shapes.png").convert("RGB")
W, H = target.size

img = render_dna(dna, (W, H))

plt.imshow(img)
plt.axis("off")
plt.show()
```

This produces an exact reconstruction of the evolved artwork.

---

## DNA Format

Each polygon is stored as:

```json
{
  "points": [[x1, y1], [x2, y2], [x3, y3], ...],
  "color": [r, g, b, a]
}
```

This preserves:

- exact polygon shape  
- orientation  
- size  
- position  
- transparency  

Nothing is lost — you can fully rebuild or transform the art later.

---

## Export to SVG

DNA can be converted into a vector image:

```python
from dna_to_svg import dna_to_svg
dna_to_svg("images/final_dna.json", (W, H), "output.svg")
```

This gives a clean, scalable version of the evolved polygons.

---

## Example

| Target Image | Evolved Output |
|--------------|----------------|
| *(your image)* | *(polygon reconstruction)* |

---

## License

MIT — free to use, modify, and extend.

---

## Acknowledgements

Inspired by classical evolutionary art projects such as the "[Evolving Mona Lisa]"(https://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/)
idea, extended here with GPU acceleration, adaptive DNA size, and customizable export options.
