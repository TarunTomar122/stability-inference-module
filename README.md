# stable-diffusion-inference-module

HOW TO RUN?

1. Open terminal in root dir
2. `PYTHONPATH=. python3 -m inference_module.example \
  --prompt "A magical forest with glowing mushrooms" \
  --negative-prompt "ugly, blurry, distorted" \
  --width 768 \
  --height 512 \
  --steps 50 \
  --guidance-scale 8.5 \
  --output "forest.png"`
