#!/usr/bin/env python3
"""Create sequential review sheets and native cut frames from the final film."""
import json, math, subprocess
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
film = ROOT / 'final/renders/no_audio_preview.mp4'
out = ROOT / 'final/qa/contact_sheets'
out.mkdir(parents=True, exist_ok=True)
probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(film)]))
duration = float(probe['format']['duration'])

def frame(at, path, scale=None):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(at), '-i', str(film), '-frames:v', '1']
    if scale: cmd += ['-vf', f'scale={scale[0]}:{scale[1]}']
    subprocess.run(cmd + [str(path)], check=True)

times = list(range(0, math.ceil(duration), 5))
for page in range(math.ceil(len(times) / 12)):
    sheet = Image.new('RGB', (1536, 732), '#091016')
    for index, at in enumerate(times[page * 12:(page + 1) * 12]):
        tmp = out / f'_composite_{at:03d}.png'
        frame(at, tmp, (384, 216))
        image = Image.open(tmp).convert('RGB')
        x, y = (index % 4) * 384, (index // 4) * 244
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x + 10, y + 218), f'{at // 60}:{at % 60:02d}', fill='#dfe9ef')
        tmp.unlink()
    sheet.save(out / f'full_composite_{page + 1:02d}.png')

for at in (8, 96, 171, 216, 261, 273):
    for name, offset in (('before', -.15), ('after', .15)):
        frame(at + offset, out / f'boundary_{at}_{name}.png')
print(f'Composite review sheets and cut frames: {out}')
