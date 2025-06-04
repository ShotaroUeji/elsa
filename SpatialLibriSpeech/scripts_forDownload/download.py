{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beeec000",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mRunning cells with 'Python 3.12.3' requires the ipykernel package.\n",
      "\u001b[1;31mInstall 'ipykernel' into the Python environment. \n",
      "\u001b[1;31mCommand: '/bin/python3 -m pip install ipykernel -U --user --force-reinstall'"
     ]
    }
   ],
   "source": [
    "import pandas as pd, requests, os\n",
    "from concurrent.futures import ThreadPoolExecutor\n",
    "from pathlib import Path\n",
    "\n",
    "BASE = \"https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1/ambisonics\"\n",
    "OUT  = Path(\"~/data/Spatial-LibriSpeech/audio\").expanduser()\n",
    "META = \"metadata.parquet\"        # すでに持っているファイル\n",
    "\n",
    "meta = pd.read_parquet(META)\n",
    "\n",
    "def fetch(row):\n",
    "    sid = f\"{row.sample_id:06d}\"\n",
    "    split = row.split            # \"train\" or \"test\"\n",
    "    url = f\"{BASE}/{sid}.flac\"\n",
    "    tgt = OUT / split / f\"{sid}.flac\"\n",
    "    if tgt.exists():             # 既にあれば skip\n",
    "        return\n",
    "    tgt.parent.mkdir(parents=True, exist_ok=True)\n",
    "    with requests.get(url, stream=True, timeout=60) as r:\n",
    "        r.raise_for_status()\n",
    "        with open(tgt, \"wb\") as f:\n",
    "            for chunk in r.iter_content(1 << 16):\n",
    "                f.write(chunk)\n",
    "\n",
    "with ThreadPoolExecutor(max_workers=8) as ex:\n",
    "    list(ex.map(fetch, [row for _, row in meta.iterrows()]))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
