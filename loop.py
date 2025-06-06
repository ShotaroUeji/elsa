-import random, json, yaml, math, sys
-from pathlib import Path
-import numpy as np, librosa, soundfile as sf
+import random, json, yaml, math, sys, hashlib, shutil
+from pathlib import Path
+import numpy as np, librosa, soundfile as sf
@@
-    foa = np.stack([W, Y, Z, X])
-
-    out_dir.mkdir(parents=True, exist_ok=True)
-
-    sf.write(out_dir/'foa.wav',  foa.T,  fs)
+    foa = np.stack([W, Y, Z, X])
+
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    # ─── Save temporary FOA (will be moved by make_pairs) ───
+    sf.write(out_dir / 'foa