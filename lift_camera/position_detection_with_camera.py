"""
Script 2: Positionsmessung
==========================
Lädt die gespeicherte Homographie ('homographie.npy') und erlaubt
das Messen von Weltkoordinaten durch Klicken ins Kamerabild.

Tasten:
  Mausklick  Position messen (Ausgabe in mm)
  C          Messpunkte löschen
  Q          Beenden

Anforderungen:
  pip install opencv-python numpy
  → Zuerst Script 1 (1_kalibrierung.py) ausführen!
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
KAMERA_INDEX   = 1                   # 0 = eingebaute Webcam
SPEICHERPFAD   = "homographie.npy"   # muss mit Script 1 übereinstimmen

# ─────────────────────────────────────────────────────────────────────────────
# Globaler Zustand
# ─────────────────────────────────────────────────────────────────────────────
messliste  = []   # [(bild_x, bild_y, welt_x, welt_y), ...]
homographie = None


def bild_zu_welt(px, py, H):
    """Transformiert Bildpunkt (Pixel) → Weltkoordinate (mm)."""
    punkt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    return cv2.perspectiveTransform(punkt, H)[0][0]


def maus_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or homographie is None:
        return
    wxy = bild_zu_welt(x, y, homographie)
    messliste.append((x, y, wxy[0], wxy[1]))
    print(f"  📍 Bild ({x:4d}, {y:4d}) px  →  Welt ({wxy[0]:8.2f}, {wxy[1]:8.2f}) mm")


def zeichne_overlay(frame):
    anzeige = frame.copy()
    h, w = anzeige.shape[:2]

    # Messpunkte einzeichnen
    for i, (bx, by, wx, wy) in enumerate(messliste):
        cv2.circle(anzeige, (bx, by), 8, (0, 60, 220), -1)
        cv2.circle(anzeige, (bx, by), 8, (255, 255, 255), 2)
        cv2.putText(anzeige, f"#{i+1}", (bx + 11, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(anzeige, f"({wx:.1f}, {wy:.1f}) mm", (bx + 11, by + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Statusbox
    status  = f"Messmodus aktiv  |  {len(messliste)} Punkt(e) gemessen"
    hinweis = "[Klick] Messen   [C] Punkte löschen   [Q] Beenden"
    cv2.rectangle(anzeige, (0, 0), (w, 58), (0, 100, 20), -1)
    cv2.putText(anzeige, status,  (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(anzeige, hinweis, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,230,200), 1, cv2.LINE_AA)

    return anzeige


def main():
    global homographie, messliste

    # ── Homographie laden ─────────────────────────────────────────────────────
    if not os.path.exists(SPEICHERPFAD):
        print(f"✗ Datei '{SPEICHERPFAD}' nicht gefunden!")
        print(f"  → Zuerst '1_kalibrierung.py' ausführen.")
        return

    homographie = np.load(SPEICHERPFAD)
    print("=" * 55)
    print("  Script 2: Positionsmessung")
    print("=" * 55)
    print(f"  ✓ Homographie geladen: {SPEICHERPFAD}")
    print(f"\n  Homographie-Matrix:\n{np.round(homographie, 5)}\n")
    print("  → Ins Kamerabild klicken um Positionen zu messen.")
    print("  → Ausgabe in mm (Ursprung = erste Schachbrettecke)\n")

    # ── Kamera öffnen ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(KAMERA_INDEX)
    if not cap.isOpened():
        print(f"✗ Kamera {KAMERA_INDEX} konnte nicht geöffnet werden!")
        return

    fenster = "Positionsmessung"
    cv2.namedWindow(fenster)
    cv2.setMouseCallback(fenster, maus_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Kein Kamerabild!")
            break

        anzeige = zeichne_overlay(frame)
        cv2.imshow(fenster, anzeige)

        taste = cv2.waitKey(1) & 0xFF

        if taste == ord('q'):
            break
        elif taste == ord('c'):
            messliste = []
            print("  Messpunkte gelöscht.")

    cap.release()
    cv2.destroyAllWindows()

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    if messliste:
        print("\n  Gemessene Punkte:")
        print(f"  {'#':>3}  {'Bild-X':>7}  {'Bild-Y':>7}  {'Welt-X (mm)':>12}  {'Welt-Y (mm)':>12}")
        print("  " + "-" * 48)
        for i, (bx, by, wx, wy) in enumerate(messliste):
            print(f"  {i+1:>3}  {bx:>7}  {by:>7}  {wx:>12.2f}  {wy:>12.2f}")

    print("\nProgramm beendet.")


if __name__ == "__main__":
    main()