# Terrain Generation and Preview

## Generate Terrain

Creates a heightfield PNG for uneven terrain in MuJoCo.

```bash
python generate_terrain.py
```

This generates `common/heightfields/terrain.png` (20m×20m terrain with smooth bumps).

## Preview Terrain

Renders a video of the humanoid on the terrain without training.

```bash
python preview_terrain.py
```

This creates `rollout.mp4` showing the humanoid on the current terrain.

---

**Note:** Run `generate_terrain.py` first, then `preview_terrain.py` to visualize before training.
