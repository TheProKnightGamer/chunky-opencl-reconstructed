"""Build a small synthetic emission pack for testing the Emissive tab / emission maps.

    make_emissive_testpack.py <minecraft client jar> <out.zip>

Marks the bright (or red) texels of a few vanilla textures as emissive, as LabPBR
_s.png alpha (254 = emits, 255 = not), plus one OptiFine _e.png overlay
(redstone_lamp_on), an all-255 _s (must be dropped) and a _n normal map (must be
ignored). crying_obsidian's rule matches nothing, so its map is empty and dropped.
Needs numpy and PIL (the harness venv has both).
"""
import zipfile, io, sys, os
import numpy as np
from PIL import Image
jar = zipfile.ZipFile(sys.argv[1])
T = 'assets/minecraft/textures/block/'
def load(name):
    return np.asarray(Image.open(io.BytesIO(jar.read(T + name + '.png'))).convert('RGBA')).astype(np.float32) / 255
def lum(a): return 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
rules = {
  'torch':            lambda a: (a[...,0] > 0.8) & (a[...,1] > 0.55) & (a[...,3] > 0),
  'glowstone':        lambda a: lum(a) > np.median(lum(a)),
  'lantern':          lambda a: (lum(a) > 0.6) & (a[...,3] > 0),
  'jack_o_lantern':   lambda a: (a[...,0] > 0.85) & (a[...,1] > 0.6),
  'sea_lantern':      lambda a: lum(a) > 0.75,
  'redstone_ore':     lambda a: (a[...,0] > 0.5) & (a[...,1] < 0.3),
  'redstone_torch':   lambda a: (a[...,0] > 0.6) & (a[...,1] < 0.4) & (a[...,3] > 0),
  'magma':            lambda a: (a[...,0] > 0.7) & (a[...,1] > 0.3),
  'crying_obsidian':  lambda a: lum(a) > 0.35,
}
out = sys.argv[2]
z = zipfile.ZipFile(out, 'w')
z.writestr('pack.mcmeta', '{"pack":{"pack_format":34,"description":"ChunkyCL synthetic LabPBR test"}}')
for name, rule in rules.items():
    a = load(name)
    m = rule(a)
    alpha = np.where(m, 254, 255).astype(np.uint8)
    rgba = np.zeros(a.shape[:2] + (4,), np.uint8); rgba[..., 0] = 128; rgba[..., 3] = alpha
    buf = io.BytesIO(); Image.fromarray(rgba, 'RGBA').save(buf, 'PNG')
    z.writestr(T + name + '_s.png', buf.getvalue())
    print(name, a.shape, 'emissive px', int(m.sum()), 'of', m.size)
# OptiFine overlay for the lit redstone lamp
z.writestr('assets/minecraft/optifine/emissive.properties', 'suffix.emissive=_e\n')
a = load('redstone_lamp_on'); m = lum(a) > 0.7
rgba = (a * 255).astype(np.uint8); rgba[..., 3] = np.where(m, 255, 0)
buf = io.BytesIO(); Image.fromarray(rgba, 'RGBA').save(buf, 'PNG')
z.writestr(T + 'redstone_lamp_on_e.png', buf.getvalue())
print('redstone_lamp_on (optifine)', int(m.sum()))
# a normal map that must be ignored, and an all-255 _s that must be dropped
buf = io.BytesIO(); Image.fromarray(np.full((16,16,4), 255, np.uint8), 'RGBA').save(buf, 'PNG')
z.writestr(T + 'stone_s.png', buf.getvalue()); z.writestr(T + 'stone_n.png', buf.getvalue())
z.close()
