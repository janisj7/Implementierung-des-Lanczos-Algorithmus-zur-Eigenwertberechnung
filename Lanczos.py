import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from scipy.sparse.linalg import svds



###########################################Initialization der Bilder und der Matrix###################################################
# 1. Daten laden (400 Bilder, jeweils 64x64 Pixel)
print("Lade Olivetti Faces...")
dataset = fetch_olivetti_faces(shuffle=True, random_state=42) #Shuffle ist klar, random state heisst immer gleich shuffeln, damit das ergebnis bleibt, reproduzierbarkeit und so.
faces = dataset.data  # Shape: (400, 4096) - jedes Bild ist bereits ein 1D-Vektor / 400 Bilder, 64 * 64 Pixel als Eintraege der Vektoren.
n_samples, n_features = faces.shape # n_samples = 400, n_features = 4096
image_shape = (64, 64)  # Shape der ursprünglichen Bilder, fuer meine Visualisierung spaeter


# 2. Daten zentrieren (Sehr wichtig für die Hauptkomponentenanalyse/PCA!)
# Wir berechnen das "Durchschnittsgesicht" und ziehen es von allen Bildern ab, damit es nicht durch den mittelwert verschoben ist.
mean_face = np.mean(faces, axis=0)
centered_faces = faces - mean_face # Abziehen des Durchschnittsgesichts von jedem Bild, damit die Daten zentriert sind. Das ist wichtig für die PCA, damit die Hauptkomponenten die Richtung der größten Varianz zeigen und nicht nur den Mittelwert.

##################################################################################################################



###########################################Lanczos Verfahren###################################################
# k=n, damit wir Lanczos scheitern sehen
k= min(centered_faces.shape) # Also einfach der Rang der nihct quadratischen Matrix centered_faces
print(f"Wir berechnen jetzt alle {k} Eigenfaces mittels Lanczos")


#symmetrische Matrix X * X^T  erzeeugen, damit Lanczos funktioniert!!
A = np.dot(centered_faces.T, centered_faces) #eine 4096 * 4096 Matrix.

# Die Orthonormalbasis für die Lanczos-Iteration
V = np.zeros((A.shape[0], k))

alphas = np.zeros(k)
betas = np.zeros(k - 1) #gibt natuerlich ein beta weniger

#Unser zufaelliger Startvektor q mit 4096 Eintraegen
q = np.random.rand(A.shape[1])
q = q / np.linalg.norm(q)

# Speichern des ersten Vektors als erste SPalte von unserer ONB V
V[:, 0] = q

# Initialisieren von q_prev und beta für die erste Iteration
q_prev = np.zeros_like(q)
beta = 0

for i in range(k):
    # 1. w = A * q berechnen
    w = np.dot(A, q)
    # 2. Orthogonalisierung gegen den vorherigen Vektor (deswegen erst ab i > 0), fuer i = 0 gibt es ja keinen.
    if i > 0:
        w = w - beta * q_prev # Orthogonalisierung gegen den vorherigen Vektor
    alpha = np.dot(q, w) # Alpha berehcnen
    alphas[i] = alpha # <--- ALPHA SPEICHERN!

    # 3. Orthogonalisierung gegen den aktuellen Vektor
    w = w - alpha * q

    # fuer naechste Iteration vorbereiten
    beta = np.linalg.norm(w) 
    q_prev = q 

    # 4. Vorzeitiger Abbruch, wenn beta = 0 bzw sehr klein ist. Hat mit dem Startvektor zu tun, falls der keinerlei anteil in richtung eines Eigenvektors hat, werden wir diesen nicht finden, da beta 0 wird.
    if beta < 1e-10: 
            print(f"Vorzeitiger Abbruch bei Iteration {i}")
            # Wir schneiden unsere Matrizen/Arrays auf die bisherige Größe zu
            V = V[:, :i+1] #Dimension anpassen
            alphas = alphas[:i+1]
            betas = betas[:i]
            break
        
    # 5. Für den nächsten Schritt vorbereiten UND speichern
    if i < k - 1:
        betas[i] = beta         # <--- BETA SPEICHERN!
        q = w / beta            # Das neue q berechnen...
        V[:, i+1] = q #i+1, da Startvektor schon in V[:, 0] gespeichert ist
     
########################################################################################################################



###########################################Tridiagonalmatrix T_m berechnen###################################################
# 1. Die Tridiagonalmatrix T aufbauen
# np.diag baut Matrizen aus Vektoren. k=0 ist die Hauptdiagonale, k=1 drüber, k=-1 drunter.
T = np.diag(alphas) + np.diag(betas, k=1) + np.diag(betas, k=-1)

# 2. Eigenwerte von T berechnen (Das sind unsere Ritz-Werte)
# Wir nutzen 'eigvalsh', da unsere Matrix T symmetrisch ist.
ritz_werte = np.linalg.eigvalsh(T)

# eigvalsh sortiert von klein nach groß. Wir drehen das um,
# weil uns die größten (dominantesten) Eigenwerte am meisten interessieren.
ritz_werte = ritz_werte[::-1]

# Vergleich mit den tatsaechlichen Eigenwerten von A. Nehmen aber X * X^T, das hat die selben EW (!=0) wie X^T * X, aber nur 400 statt 4096
A_small = np.dot(centered_faces, centered_faces.T)
true_eigenvalues = np.linalg.eigvalsh(A_small)
true_eigenvalues = true_eigenvalues[::-1]

##########################################################################################################################



###########################################Plotten der Ritz-Werte & richtigen Eigenwerte###################################################
plt.figure(figsize=(12, 6))

n_plot = min(k, 100) # Wir plotten nur die ersten 100 Ritz-Werte, um die Übersicht zu behalten.

# 1. Die wahren Eigenwerte als HORIZONTALE Linien (Niveaus) zeichnen
# Wir zeichnen nur die ersten paar (z.B. 10), da es unten sonst zu unübersichtlich wird.
num_true_to_plot = 10 
for idx, val in enumerate(true_eigenvalues[:num_true_to_plot]):
    # Nur bei der ersten Linie ein Label für die Legende setzen
    label = 'Wahre Eigenwerte (Diskrete Niveaus)' if idx == 0 else ""
    plt.axhline(y=val, color='blue', linestyle='--', alpha=0.4, label=label)

# 2. Unsere Lanczos-Eigenwerte (Rote Punkte)
plt.plot(ritz_werte[:n_plot], 'ro', markersize=6, label='Lanczos Ritz-Werte (Aus Matrix T)')

plt.title("Warum Lanczos nicht funktioniert", fontsize=16)
plt.ylabel("Größe des Eigenwerts", fontsize=12)
plt.xlabel("Index (Die k größten Ritz-Werte)", fontsize=12)

# Legende und Grid
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)

# Logarithmische Y-Achse
plt.yscale('log')
plt.show()

##############################################################################################################


































# # # # svds wendet iterativ Matrix-Vektor-Multiplikationen an.
# # # U, S, Vt = svds(centered_faces, k=k)

# # # SciPy gibt die Werte in aufsteigender Reihenfolge zurück. 
# # # Wir drehen sie um, damit die wichtigsten (größter Eigenwert) vorne stehen.
# # S = S[::-1]
# # Vt = Vt[::-1]

# eigenfaces = Vt  # Die Zeilen von Vt sind unsere gesuchten Eigenfaces

# # 4. Visualisierung für die Präsentation
# def plot_gallery(title, images, n_col=4, n_row=3, cmap=plt.cm.gray):
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     plt.suptitle(title, size=16)
#     for i, comp in enumerate(images):
#         plt.subplot(n_row, n_col, i + 1)
#         # Werte für die Darstellung skalieren
#         vmax = max(comp.max(), -comp.min())
#         plt.imshow(comp.reshape(image_shape), cmap=cmap,
#                    interpolation='nearest', vmin=-vmax, vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#     plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
#     plt.show()

# # Durchschnittsgesicht anzeigen (als einzelnes Bild)
# plt.figure(figsize=(3, 3))
# plt.title("Das Durchschnittsgesicht")
# plt.imshow(mean_face.reshape(image_shape), cmap=plt.cm.gray)
# plt.axis("off")
# plt.show()

# # Die berechneten Eigenfaces anzeigen
# plot_gallery(f"Die Top {k} Eigenfaces (Berechnet mit Lanczos-basiertem SVD)", eigenfaces)
