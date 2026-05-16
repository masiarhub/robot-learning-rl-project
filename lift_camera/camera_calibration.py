"""
Script 1: Kalibrierung
======================
Schachbrett vor die Kamera halten → [LEERTASTE] → Homographie wird
berechnet und als 'homographie.npy' gespeichert.

Tasten:
  LEERTASTE  Homographie berechnen & speichern
  R          Nochmal versuchen
  Q          Beenden

Anforderungen:
  pip install opencv-python numpy
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCHACHBRETT_SPALTEN = 8       # innere Ecken horizontal (Felder - 1)
SCHACHBRETT_ZEILEN  = 6       # innere Ecken vertikal   (Felder - 1)
FELD_GROESSE_MM     = 25.0    # reale Feldgrösse in mm
KAMERA_INDEX        = 0       # 0 = eingebaute Webcam
SPEICHERPFAD        = "homographie.npy"

# ─────────────────────────────────────────────────────────────────────────────
# Subpixel-Kriterien & Weltkoordinaten
# ─────────────────────────────────────────────────────────────────────────────
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WELT_ECKEN = np.zeros((SCHACHBRETT_SPALTEN * SCHACHBRETT_ZEILEN, 2), dtype=np.float32)
WELT_ECKEN[:, :] = np.mgrid[0:SCHACHBRETT_SPALTEN, 0:SCHACHBRETT_ZEILEN].T.reshape(-1, 2)
WELT_ECKEN *= FELD_GROESSE_MM


def berechne_reprojektionsfehler(bild_pts, welt_pts, H):
    fehler = []
    for (bx, by), (wx, wy) in zip(bild_pts, welt_pts):
        p = cv2.perspectiveTransform(np.array([[[bx, by]]], dtype=np.float32), H)[0][0]
        fehler.append(np.linalg.norm(p - np.array([wx, wy])))
    return float(np.mean(fehler))


def zeichne_overlay(frame, gefunden, ecken):
    anzeige = frame.copy()
    h, w = anzeige.shape[:2]

    if gefunden and ecken is not None:
        cv2.drawChessboardCorners(anzeige, (SCHACHBRETT_SPALTEN, SCHACHBRETT_ZEILEN), ecken, gefunden)
        status  = f"Brett erkannt ({SCHACHBRETT_SPALTEN}x{SCHACHBRETT_ZEILEN})  –  [LEERTASTE] kalibrieren"
        farbe   = (0, 110, 0)
    else:
        status  = "Suche Schachbrett ..."
        farbe   = (30, 30, 140)

    hinweis = f"Feld: {FELD_GROESSE_MM:.0f} mm  |  [R] Reset  [Q] Beenden"
    cv2.rectangle(anzeige, (0, 0), (w, 58), farbe, -1)
    cv2.putText(anzeige, status,  (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(anzeige, hinweis, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,230,200), 1, cv2.LINE_AA)
    return anzeige


def main():
    cap = cv2.VideoCapture(KAMERA_INDEX)
    if not cap.isOpened():
        print(f"✗ Kamera {KAMERA_INDEX} konnte nicht geöffnet werden!")
        return

    fenster = "Kalibrierung – Schachbrett"
    cv2.namedWindow(fenster)

    print("=" * 55)
    print("  Script 1: Kalibrierung")
    print("=" * 55)
    print(f"  Brett:  {SCHACHBRETT_SPALTEN} × {SCHACHBRETT_ZEILEN} innere Ecken")
    print(f"  Feld:   {FELD_GROESSE_MM} mm")
    print()
    print("  → Schachbrett vor die Kamera halten")
    print("  → Wenn grüne Ecken erscheinen: [LEERTASTE] drücken")
    print()

    brett_groesse  = (SCHACHBRETT_SPALTEN, SCHACHBRETT_ZEILEN)
    letzte_ecken   = None
    brett_gefunden = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Kein Kamerabild!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gefunden, ecken = cv2.findChessboardCorners(
            gray, brett_groesse,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if gefunden:
            ecken = cv2.cornerSubPix(gray, ecken, (11, 11), (-1, -1), CRITERIA)
            letzte_ecken   = ecken
            brett_gefunden = True
        else:
            brett_gefunden = False

        anzeige = zeichne_overlay(frame, brett_gefunden, letzte_ecken)
        cv2.imshow(fenster, anzeige)

        taste = cv2.waitKey(1) & 0xFF

        if taste == ord('q'):
            break

        elif taste == ord('r'):
            letzte_ecken   = None
            brett_gefunden = False
            print("  ↺ Reset")

        elif taste == ord(' '):
            if not brett_gefunden or letzte_ecken is None:
                print("  ⚠ Kein Brett erkannt – bitte Brett zeigen und erneut versuchen.")
                continue

            ecken_2d = letzte_ecken.reshape(-1, 2).astype(np.float32)
            H, maske = cv2.findHomography(ecken_2d, WELT_ECKEN, cv2.RANSAC, 3.0)

            if H is None:
                print("  ✗ Homographie-Berechnung fehlgeschlagen – erneut versuchen.")
                continue

            fehler = berechne_reprojektionsfehler(ecken_2d, WELT_ECKEN, H)
            np.save(SPEICHERPFAD, H)

            print(f"\n  ✓ Homographie berechnet ({SCHACHBRETT_SPALTEN * SCHACHBRETT_ZEILEN} Punkte, RANSAC)")
            print(f"  Rückprojektionsfehler: {fehler:.3f} mm", end="")
            print("  ✓ sehr gut" if fehler < 1.0 else ("  ✓ gut" if fehler < 3.0 else "  ⚠ hoch"))
            print(f"  💾 Gespeichert: {SPEICHERPFAD}")
            print(f"\n  Homographie-Matrix:\n{np.round(H, 5)}\n")
            print("  → Kalibrierung abgeschlossen. Script 2 starten zum Messen.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()