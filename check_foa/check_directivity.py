import numpy as np, pyroomacoustics as pra, math, matplotlib.pyplot as plt

room = pra.ShoeBox([3,3,3], fs=16000, max_order=0)
center = np.array([1.5,1.5,1.5])

r=0.05; v=r/math.sqrt(3)
tet = np.array([[ v,  v,  v],
                [ v, -v, -v],
                [-v,  v, -v],
                [-v, -v,  v]], dtype=float).T

dirs=[]
for x,y,z in tet.T:
    az  = math.degrees(math.atan2(y,x)) % 360
    col = math.degrees(math.acos(z/r))
    dirs.append(pra.directivities.CardioidFamily(
        orientation=pra.directivities.DirectionVector(
            azimuth=az, colatitude=col, degrees=True),
        p=0.5, gain=1.0))

pos = center[:,None] + tet
mics = pra.MicrophoneArray(pos, room.fs, directivity=dirs)
room.add_microphone_array(mics)

fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(*pos, c='b', s=60)

for p, card in zip(pos.T, dirs):
    u = card._orientation.unit_vector  # エラーにならない
    ax.quiver(*p, *u, length=0.03, color='cyan', normalize=True)

plt.title("Tetrahedral array – orientation arrows")
plt.show()