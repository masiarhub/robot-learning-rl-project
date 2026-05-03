import random
import keyboard


def generate_points() -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Generates two random (x, y) tuples where:
    - Both x and y are divisible by 4
    - x is in the range [-4, 40]
    - y is in the range [-40, 40]
    - Euclidean distance from origin >= 12 and < 42
    - Neither point is (0, 0)
    - Distance between p1 and p2 >= 10
    """
    x_values = range(-4, 41, 4)   # [-4, 0, 4, 8, ..., 40]
    y_values = range(-40, 41, 4)  # [-40, -36, ..., 0, ..., 36, 40]

    valid_points = [
        (x, y) for x in x_values for y in y_values
        if (x, y) != (0, 0) and 12 <= (x**2 + y**2) ** 0.5 < 42
    ]

    while True:
        p1 = random.choice(valid_points)
        p2 = random.choice(valid_points)

        distance_p1_p2 = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
        if distance_p1_p2 >= 10:
            return p1, p2


if __name__ == "__main__":
    print("Press Enter to generate a point, Esc to exit\n")

    point_count = 0
    while True:
        event = keyboard.read_event()
        if event.event_type == 'down':
            if event.name == 'enter':
                p1, p2 = generate_points()
                point_count += 1
                print(f"  {point_count:2}. ({p1[0]:4}, {p1[1]:4})    ({p2[0]:4}, {p2[1]:4})")
            elif event.name == 'esc':
                print("\nExiting...")
                break
